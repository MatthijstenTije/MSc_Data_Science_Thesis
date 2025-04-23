import os
import json
import logging
import glob
from collections import Counter
import ollama
from pydantic import BaseModel, conlist
from typing import List

# === Structured Output Schema ===
from pydantic import BaseModel, validator
from typing import List

class SentenceEntry(BaseModel):
    word: str
    sentence_type: str
    sentence: str

class OutputSchema(BaseModel):
    sentences: List[SentenceEntry]

# === Setup Logging Configuration ===
os.makedirs("Notebooks/Phase_02/Test/logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG, 
    format="[%(asctime)s] %(levelname)s: %(message)s",
    filename="Notebooks/Phase_02/Test/logs/main.log",
    filemode="a"
)
logger = logging.getLogger(__name__)
logger.debug("Logger initialized. Script started.")

# === Define Word Lists as Python Variables ===
male_nouns = ["man", "mannen", "jongen", "jongens", "heer", "heren", "vent"]
female_nouns = ["vrouw", "vrouwen", "meisje", "meisjes", "dame", "dames", "mevrouw"]

male_adjs = [
    "corrupt", "onoverwinnelijk", "plaatsvervangend", "impopulair", "goddeloos",
    "incompetent", "misdadig", "bekwaam", "sadistisch", "gewetenloos", 
    "steenrijk", "vooraanstaand", "voortvluchtig", "geniaal", "planmatig", "bekwaamheid","genialiteit"
]

female_adjs = [
    "blond", "beeldschoon", "bloedmooie", "donkerharig", "ongehuwd",
    "kinderloos","glamoureus", "beeldig", "sensueel", "platinablond", 
    "voorlijk", "feministisch", "stijlvol", "tuttig", "rimpelig"]

# === Parameter Setup ===
PROMPT_FOLDER = "Notebooks/Phase_02/prompts"
prompt_files = glob.glob(os.path.join(PROMPT_FOLDER, "*.txt"))

if not prompt_files:
    logger.error(f"No prompt files found in {PROMPT_FOLDER}")
    raise FileNotFoundError(f"No prompt files found in {PROMPT_FOLDER}")
logger.info(f"Found {len(prompt_files)} prompt files.")

# List of models to run.
MODELS = ["llama3-chatqa:8b","llama3:text","llama3:8b","llama2-uncensored"]

# Temperature values.
TEMPERATURES = [0.5,0.75,1,1.25,1.5]

# Set a single seed.
SEED_START = 42

# === Initialize the Ollama Client ===
client = ollama.Client()
logger.info("Initialized Ollama client.")


# === Helper: Prepare Prompt with Dynamic Ordering ===
def prepare_prompt(base_prompt: str, current_sentences: List[dict],
                   noun_words: List[str], adj_words: List[str]) -> str:
    """
    Returns a modified prompt with dynamic ordering of nouns and adjectives.
    The ordering is computed based on usage counts from the current_sentences,
    but only the ordered list is inserted (no count values are shown).
    Assumes that base_prompt contains placeholders: {noun_order} and {adjective_order}.
    """
    if current_sentences:
        # Count appearances for adjectives only
        adj_counter = Counter(entry["word"] for entry in current_sentences if entry["word"] in adj_words)
        ordered_adjs = sorted(adj_words, key=lambda w: adj_counter.get(w, 0))
    else:
        ordered_adjs = adj_words

    # Nouns remain in their original order (or similar logic could be applied)
    ordered_nouns = noun_words

    # Create comma-separated strings and insert into prompt placeholders.
    noun_order_str = ", ".join(ordered_nouns)
    adj_order_str = ", ".join(ordered_adjs)
    modified_prompt = base_prompt.format(noun_order=noun_order_str, adjective_order=adj_order_str)
    return modified_prompt

# === Read the Base Prompt Content ===
for PROMPT_FILE in prompt_files:
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as pf:
            base_prompt = pf.read()
        logger.info(f"Successfully read the prompt from {PROMPT_FILE}")
    except Exception as e:
        logger.exception(f"Failed to read the prompt file {PROMPT_FILE}: {e}")
        raise

    # === Extract Gender Information from Prompt File Name ===
    # Expected filenames: "prompt_maleNouns-maleAdjs.txt" or "prompt_femaleNouns-femaleAdjs.txt"
    prompt_filename = os.path.basename(PROMPT_FILE)
    file_without_ext = os.path.splitext(prompt_filename)[0]
    parts = file_without_ext.replace("prompt_", "").split("-")
    noun_group = parts[0]       # e.g. "maleNouns" or "femaleNouns"
    adjective_group = parts[1]  # e.g. "maleAdjs" or "femaleAdjs"

    noun_gender = "male" if "male" in noun_group.lower() else "female"
    adjective_gender = "male" if "male" in adjective_group.lower() else "female"
    logger.debug(f"Extracted noun_gender: {noun_gender} and adjective_gender: {adjective_gender} from the prompt filename.")

    # Set expected words based on gender.
    original_nouns = male_nouns if noun_gender == "male" else female_nouns
    original_adjs = male_adjs if adjective_gender == "male" else female_adjs

    def run_model(client, model: str, prompt: str, temp: float, seed: int):
        """
        Executes the model call and retrieves the structured output.
        Returns a tuple with:
        - structured_output: list of sentences (as dictionaries)
        - errors: list of error messages, if any
        """
        logger.info(f"Running model '{model}' with temperature={temp} and seed={seed}")
        logger.debug(f"Prompt sent to model {prompt}...")
        try:
            response = client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                format=OutputSchema.model_json_schema(),  # Enforced output schema
                options={"temperature": temp, "seed": seed}
            )
            logger.debug(f"Received response from model '{model}': {response}")
        except Exception as e:
            error_msg = f"Model call failed for {model}, temperature={temp}, seed={seed}: {e}"
            logger.error(error_msg)
            return None, [str(e)]
        
        errors = []
        try:
            data = OutputSchema.model_validate_json(response['message']['content'])
            structured_output = [entry.dict() for entry in data.sentences]
            # Check if exactly 15 sentences are returned:
            if len(structured_output) != 15:
                err_msg = f"Expected exactly 15 sentences, but got {len(structured_output)}"
                errors.append(err_msg)
                logger.error(err_msg)
            else:
                logger.info(f"Successfully parsed 15 sentences from model '{model}'.")
        except Exception as e:
            error_msg = f"JSON validation failed for {model}, temperature={temp}, seed={seed}: {e}"
            logger.error(error_msg)
            structured_output = []
            errors.append(str(e))
        
        return structured_output, errors

    # === Main Aggregation Loop ===
    # Target: each expected word should appear exactly 10 times.
    TARGET_COUNT_PER_WORD = 1
    TOTAL_TARGET_SENTENCES = TARGET_COUNT_PER_WORD * (len(original_adjs))
    
    for model in MODELS:
        # Create a model-specific log directory
        model_dir = os.path.join("Notebooks/Phase_02/Test/logs", model.replace(":", "_"))
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created log directory for model '{model}': {model_dir}")
        
        for temp in TEMPERATURES:
            expected_nouns = original_nouns[:]  # Copy in case it ever gets changed
            expected_adjs = original_adjs[:]  # still used for ordering
            expected_adjs_set = set(expected_adjs)  # for fast look
            all_sentences = []       # Collected valid sentences.
            aggregated_errors = []   # Collect encountered errors.
            run_count = 0            # Counter for runs.
            # Initialize word counter to track counts of expected words.
            word_counter = Counter({word: 0 for word in expected_adjs})

            logger.info(f"Starting aggregation for model '{model}' with temperature {temp}.")
            # Continue until all expected words have reached the target count.
            while sum(word_counter.values()) < TOTAL_TARGET_SENTENCES:
                current_seed = SEED_START + run_count
                run_count += 1

                modified_prompt = prepare_prompt(base_prompt, all_sentences, expected_nouns, expected_adjs)
                structured_output, errors = run_model(client, model, modified_prompt, temp, current_seed)
                if errors:
                    aggregated_errors.extend(errors)
                    logger.warning(f"Run {run_count}: Encountered errors: {errors}")

                logger.info(f"Before processing run {run_count}, current word counts: {dict(word_counter)}")
                if structured_output:
                    for entry in structured_output:
                        word = entry["word"]
                        # Skip words that are not in the expected list.
                        if word not in expected_adjs_set:
                            logger.debug(f"Skipping unexpected word '{word}' from output.")
                            continue
                        # Append only if the target count has not been reached.
                        if word_counter[word] < TARGET_COUNT_PER_WORD:
                            all_sentences.append(entry)
                            word_counter[word] += 1
                            # Remove word from expected_adjs once it hits the target count
                            if word_counter[word] >= TARGET_COUNT_PER_WORD:
                                try:
                                    expected_adjs.remove(word)
                                    expected_adjs_set.remove(word)  # update the set too
                                    logger.info(f"Word '{word}' reached target and was removed from adjective list.")
                                except ValueError:
                                    logger.warning(f"Tried to remove word '{word}', but it was already removed.")
                        else:
                            logger.warning(
                                "Skipping sentence with word '%s' because its count (%s) reached the target (%s).",
                                word, word_counter[word], TARGET_COUNT_PER_WORD
                            )
                logger.info(f"After run {run_count}, word counts: {dict(word_counter)}; total sentences: {len(all_sentences)}")
                
                # If all expected words have reached the target, stop early.
                if all(word_counter[w] >= TARGET_COUNT_PER_WORD for w in expected_adjs):
                    logger.info("All expected words have reached the target count.")
                    break

            # Trim to exactly the TOTAL_TARGET_SENTENCES.
            all_sentences = all_sentences[:TOTAL_TARGET_SENTENCES]
            logger.info(f"Final aggregated sentence count for model '{model}' with temperature {temp}: {len(all_sentences)}")

            # Create a log filename with prompt and temperature information.
            log_filename = f"{noun_group}-{adjective_group}_temp{temp}.jsonl"
            log_file_path = os.path.join(model_dir, log_filename)
            
            # Build aggregated log data.
            log_data = {
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
            
            try:
                with open(log_file_path, "a", encoding="utf-8") as lf:
                    json.dump(log_data, lf, ensure_ascii=False)
                    lf.write("\n")
                logger.info(f"Logged aggregated output to {log_file_path} for model '{model}' with temperature {temp}")
            except Exception as e:
                logger.exception(f"Failed to write log file {log_file_path}: {e}")

    logger.info("Completed aggregating outputs for all models and temperatures.")