import os
import json
import logging
import glob
import asyncio
from collections import Counter
from pydantic import BaseModel
from typing import List
from ollama import AsyncClient

# === Structured Output Schema ===
class SentenceEntry(BaseModel):
    word: str
    sentence_type: str
    sentence: str

class OutputSchema(BaseModel):
    sentences: List[SentenceEntry]

# === Setup Logging Configuration ===
os.makedirs("Notebooks/Phase_02/logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    filename="Notebooks/Phase_02/logs/main.log",
    filemode="a"
)
logger = logging.getLogger(__name__)
logger.debug("Logger initialized. Script started.")

# === Define Word Lists ===
male_nouns = ["man", "mannen", "jongen", "jongens", "heer", "heren", "vent"]
female_nouns = ["vrouw", "vrouwen", "meisje", "meisjes", "dame", "dames", "mevrouw"]

male_adjs = [
    "corrupt", "onoverwinnelijk", "plaatsvervangend", "impopulair", "goddeloos",
    "incompetent", "misdadig", "bekwaam", "sadistisch", "gewetenloos",
    "steenrijk", "vooraanstaand", "voortvluchtig", "geniaal", "planmatig"
]

female_adjs = [
    "blond", "beeldschoon", "bloedmooie", "donkerharig", "ongehuwd",
    "kinderloos", "glamoureus", "beeldig", "sensueel", "platinablond",
    "voorlijk", "feministisch", "stijlvol", "tuttig", "rimpelig"
]

PROMPT_FOLDER = "Notebooks/Phase_02/prompts"
prompt_files = glob.glob(os.path.join(PROMPT_FOLDER, "*.txt"))

MODELS = ["llama3-chatqa:8b", "llama3:text", "llama3:8b", "llama2-uncensored"]
TEMPERATURES = [0.5, 0.75, 1, 1.25, 1.5]
SEED_START = 42
TARGET_COUNT_PER_WORD = 15

# === Helper ===
def prepare_prompt(base_prompt: str, current_sentences: List[dict], noun_words: List[str], adj_words: List[str]) -> str:
    adj_counter = Counter(entry["word"] for entry in current_sentences if entry["word"] in adj_words)
    ordered_adjs = sorted(adj_words, key=lambda w: adj_counter.get(w, 0))
    noun_order_str = ", ".join(noun_words)
    adj_order_str = ", ".join(ordered_adjs)
    return base_prompt.format(noun_order=noun_order_str, adjective_order=adj_order_str)

async def run_model(client, model: str, prompt: str, temp: float, seed: int):
    try:
        response = await client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            format=OutputSchema.model_json_schema(),
            options={"temperature": temp, "seed": seed}
        )
    except Exception as e:
        return None, [str(e)]

    try:
        data = OutputSchema.model_validate_json(response['message']['content'])
        structured_output = [entry.dict() for entry in data.sentences]
        errors = []
        if len(structured_output) != 15:
            errors.append(f"Expected exactly 15 sentences, got {len(structured_output)}")
        return structured_output, errors
    except Exception as e:
        return [], [str(e)]

async def process_model_temperature(client, semaphore, model, temp, base_prompt, noun_group, adjective_group, original_nouns, original_adjs, noun_gender, adjective_gender):
    async with semaphore:
        expected_adjs = original_adjs[:]
        all_sentences = []
        aggregated_errors = []
        run_count = 0
        word_counter = Counter({word: 0 for word in expected_adjs})
        TOTAL_TARGET_SENTENCES = TARGET_COUNT_PER_WORD * len(original_adjs)

        while sum(word_counter.values()) < TOTAL_TARGET_SENTENCES:
            current_seed = SEED_START + run_count
            run_count += 1
            prompt = prepare_prompt(base_prompt, all_sentences, original_nouns, expected_adjs)
            structured_output, errors = await run_model(client, model, prompt, temp, current_seed)
            if errors:
                aggregated_errors.extend(errors)
            if structured_output:
                for entry in structured_output:
                    word = entry["word"]
                    if word not in expected_adjs:
                        continue
                    if word_counter[word] < TARGET_COUNT_PER_WORD:
                        all_sentences.append(entry)
                        word_counter[word] += 1
                        if word_counter[word] >= TARGET_COUNT_PER_WORD:
                            try:
                                expected_adjs.remove(word)
                            except ValueError:
                                pass

        all_sentences = all_sentences[:TOTAL_TARGET_SENTENCES]

        return {
            "model": model,
            "temperature": temp,
            "log_filename": f"{noun_group}-{adjective_group}_temp{temp}.jsonl",
            "log_dir": os.path.join("Notebooks/Phase_02/logs", model.replace(":", "_")),
            "log_data": {
                "model": model,
                "temperature": temp,
                "total_runs": run_count,
                "seeds_used": list(range(SEED_START, SEED_START + run_count)),
                "aggregated_output": all_sentences,
                "total_sentences": len(all_sentences),
                "parse_errors": aggregated_errors,
                "noun_gender": noun_gender,
                "adjective_gender": adjective_gender,
                "final_word_counts": dict(word_counter)
            }
        }

async def main():
    if not prompt_files:
        logger.error(f"No prompt files found in {PROMPT_FOLDER}")
        return

    client = AsyncClient()
    semaphore = asyncio.Semaphore(4)
    tasks = []

    for PROMPT_FILE in prompt_files:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as pf:
            base_prompt = pf.read()

        prompt_filename = os.path.basename(PROMPT_FILE)
        file_without_ext = os.path.splitext(prompt_filename)[0]
        parts = file_without_ext.replace("prompt_", "").split("-")
        noun_group, adjective_group = parts
        noun_gender = "male" if "male" in noun_group.lower() else "female"
        adjective_gender = "male" if "male" in adjective_group.lower() else "female"
        original_nouns = male_nouns if noun_gender == "male" else female_nouns
        original_adjs = male_adjs if adjective_gender == "male" else female_adjs

        for model in MODELS:
            for temp in TEMPERATURES:
                tasks.append(process_model_temperature(
                    client, semaphore, model, temp, base_prompt,
                    noun_group, adjective_group,
                    original_nouns, original_adjs,
                    noun_gender, adjective_gender
                ))

    results = await asyncio.gather(*tasks)

    for result in results:
        if result is None:
            continue
        os.makedirs(result["log_dir"], exist_ok=True)
        log_path = os.path.join(result["log_dir"], result["log_filename"])
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(result["log_data"], f, ensure_ascii=False)
            f.write("\n")
        logger.info(f"Wrote log to {log_path}")

if __name__ == "__main__":
    asyncio.run(main())

