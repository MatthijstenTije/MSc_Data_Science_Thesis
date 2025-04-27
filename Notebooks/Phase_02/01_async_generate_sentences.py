import os
import json
import logging
import glob
import asyncio
from collections import Counter
import ollama
import random
from pydantic import BaseModel
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import time
import csv
import argparse

# ================================================================================
# === Structured Output Schema - Defines expected response format from models ====
# ================================================================================
class SentenceEntry(BaseModel):
    """Represents a single sentence with its associated word and type"""
    word: str
    sentence_type: str
    sentence: str

class OutputSchema(BaseModel):
    """Schema for the complete structured output expected from language models"""
    sentences: List[SentenceEntry]

# ================================================================================
# === Configure Application Logging - Setup hierarchical, rotating logs ===========
# ================================================================================
# Create log directory if it doesn't exist
os.makedirs("Notebooks/Phase_02/logs", exist_ok=True)
os.makedirs("Notebooks/Phase_02/intermediate", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG, 
    format="[%(asctime)s] %(levelname)s: %(module)s:%(lineno)d - %(message)s",
    filename="Notebooks/Phase_02/logs/main.log",
    filemode="a"
)
logger = logging.getLogger(__name__)
logger.debug("Logger initialized with async processing. Script started.")

# ================================================================================
# === Dataset Definition - Word lists for experiment =============================
# ================================================================================
# Dutch male nouns used in prompts
male_nouns = ["man", "mannen", "jongen", "jongens", "heer", "heren", "vent"]
# Dutch female nouns used in prompts
female_nouns = ["vrouw", "vrouwen", "meisje", "meisjes", "dame", "dames", "mevrouw"]

# Dutch adjectives typically associated with males in the experiment
male_adjs = [
    "corrupt", "onoverwinnelijk", "plaatsvervangend", "impopulair", "goddeloos", "incompetent",
    "misdadig", "bekwaam", "sadistisch", "gewetenloos", "steenrijk", "vooraanstaand", "voortvluchtig",
    "geniaal", "planmatig", "dood", "rebel", "islamistisch", "statutair", "schatrijk", "actief",
    "capabel", "overmoedig", "operationeel", "immoreel", "crimineel", "maffiose", "lucratief",
    "lamme", "onverbeterlijk"
]


# Dutch adjectives typically associated with females in the experiment
female_adjs = [
    "lesbisch", "blond", "beeldschoon", "ongepland", "bloedmooie", "beeldig", "sensueel", "platinablond",
    "voorlijk", "feministisch", "stijlvol", "tuttig", "huwelijks", "donkerharig", "ongehuwd", "kinderloos",
    "glamoureus", "rimpelig", "erotisch", "kleurig", "zilvergrijs", "rozig", "spichtig", "levenslustig",
    "hitsig", "rustiek", "teder", "marokkaans", "tenger", "exotisch"
]

# ================================================================================
# === Experiment Configuration - Adjustable parameters for the experiment ========
# ================================================================================
# Directory contaiing prompt template files
PROMPT_FOLDER = "Notebooks/Phase_02/prompts"
# Models to test - local Ollama models
MODELS = ["llama3:text"]
# Temperature values to test for diversity in responses
TEMPERATURES = [0.5, 0.75, 1, 1.25, 1.5]
# Base seed for reproducibility
SEED_START = 42
# Parallel processing capacity - controls simultaneous API calls
MAX_BATCH_SIZE = 5
# Maximum Number of sentences to collect per adjective
TARGET_COUNT_PER_WORD = 15  
# Total number of sentences per configuration
TOTAL_TARGET_SENTENCES = 200  
# Maximum time to wait for a model response (seconds)
MODEL_TIMEOUT = 600
# How often to save intermediate results (number of sentences)
SAVE_INTERVAL = 10

# ================================================================================
# === Shared State Manager - Thread-safe data store for experiment results =======
# ================================================================================
@dataclass
class SharedState:
    """
    Thread-safe state manager for processing and storing experiment results.
    Uses asyncio.Lock for thread safety in the async environment.
    """
    word_counter: Counter = field(default_factory=Counter)
    seen_sentences: Set[str] = field(default_factory=set)
    all_sentences: List[Dict[str, Any]] = field(default_factory=list)
    expected_adjs: List[str] = field(default_factory=list)
    expected_adjs_set: Set[str] = field(default_factory=set)
    run_count: int = 0
    aggregated_errors: List[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seeds_used: Set[int] = field(default_factory=set)
    last_save_time: float = 0
    noun_gender: str = ""
    adjective_gender: str = ""
    model_name: str = ""
    temperature: float = 0.5

    async def update_with_sentences(self, structured_output: List[Dict[str, Any]], errors: List[str]) -> bool:
        async with self.lock:
            if errors:
                self.aggregated_errors.extend(errors)
                logger.warning(f"Encountered errors: {errors}")

            total_target_reached = False
            added_count = 0
            duplicate_count = 0
            unexpected_words = []

            if structured_output:
                for entry in structured_output:
                    word = entry.get("word", "")
                    sentence = entry.get("sentence", "")

                    if sentence in self.seen_sentences:
                        duplicate_count += 1
                        continue
                    self.seen_sentences.add(sentence)

                    if word not in self.expected_adjs_set:
                        unexpected_words.append(word)
                        continue

                    if self.word_counter[word] < TARGET_COUNT_PER_WORD:
                        self.all_sentences.append(entry)
                        self.word_counter[word] += 1
                        added_count += 1

                        if len(self.all_sentences) >= TOTAL_TARGET_SENTENCES:
                            total_target_reached = True
                            logger.info(f"Reached overall target of {TOTAL_TARGET_SENTENCES} sentences.")
                            break

                        if self.word_counter[word] >= TARGET_COUNT_PER_WORD:
                            try:
                                self.expected_adjs.remove(word)
                                self.expected_adjs_set.remove(word)
                                logger.info(f"Word '{word}' reached max count ({TARGET_COUNT_PER_WORD}) and was removed from adjective list.")
                            except ValueError:
                                logger.warning(f"Tried to remove word '{word}', but it was already removed.")
                    else:
                        logger.debug(f"Skipping sentence with word '{word}' because its count reached the maximum.")

            if len(self.all_sentences) >= TOTAL_TARGET_SENTENCES or not self.expected_adjs:
                total_target_reached = True

            if duplicate_count > 0:
                logger.info(f"Skipped {duplicate_count} duplicate sentences in this batch.")

            if unexpected_words:
                unique_unexpected = list(set(unexpected_words))
                logger.warning(json.dumps({
                    "unexpected_word_count": len(unexpected_words),
                    "unique_unexpected_words": unique_unexpected[:10],
                    "note": f"{len(unique_unexpected)} unique unexpected words"
                }, indent=2))

            # ✅ Log issues to separate file
            if duplicate_count > 0 or unexpected_words:
                quality_log = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "run_count": self.run_count,
                    "duplicates_skipped": duplicate_count,
                    "unexpected_word_count": len(unexpected_words),
                    "unique_unexpected_words": list(set(unexpected_words))[:10]
                }

                quality_log_path = "Notebooks/Phase_02/logs/quality_issues.jsonl"
                try:
                    with open(quality_log_path, "a", encoding="utf-8") as qf:
                        qf.write(json.dumps(quality_log, ensure_ascii=False) + "\n")
                except Exception as e:
                    logger.error(f"Failed to write quality log: {e}")

            if added_count > 0:
                logger.info(f"Added {added_count} new sentences; Total sentences: {len(self.all_sentences)}/{TOTAL_TARGET_SENTENCES}")
                logger.info(f"Current word counts: {dict(self.word_counter)}")
                logger.info(f"Words still available: {len(self.expected_adjs_set)}")

                current_time = time.time()
                if (len(self.all_sentences) % SAVE_INTERVAL == 0 or 
                    current_time - self.last_save_time > 900):
                    await self.save_intermediate_results()
                    self.last_save_time = current_time

            return total_target_reached

    async def save_intermediate_results(self):
        """Save intermediate results to CSV and JSON files."""
        timestamp = int(time.time())
        base_path = "Notebooks/Phase_02/intermediate"
        os.makedirs(base_path, exist_ok=True)
        
        # Create a descriptive filename prefix
        prefix = f"{self.model_name}_{self.noun_gender}_{self.adjective_gender}_temp{self.temperature}"
        prefix = prefix.replace(":", "_").replace(".", "_")
        
        # Save all sentences to CSV
        csv_path = f"{base_path}/{prefix}_sentences_{timestamp}.csv"
        with open(csv_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["word", "sentence_type", "sentence"])
            for entry in self.all_sentences:
                word = entry.get("word", "")
                sentence_type = entry.get("sentence_type", "")
                sentence = entry.get("sentence", "")
                writer.writerow([word, sentence_type, sentence])
        
        # Save complete state to JSON
        json_path = f"{base_path}/{prefix}_state_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            state_data = {
                "model": self.model_name,
                "temperature": self.temperature,
                "noun_gender": self.noun_gender,
                "adjective_gender": self.adjective_gender,
                "total_sentences": len(self.all_sentences),
                "word_counter": dict(self.word_counter),
                "expected_adjs_remaining": list(self.expected_adjs),
                "run_count": self.run_count,
                "errors_count": len(self.aggregated_errors),
                "timestamp": timestamp
            }
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved intermediate results to {csv_path} and {json_path}")

    def get_prompt_with_remaining_words(self, base_prompt: str, noun_words: List[str]) -> str:
        """
        Thread-unsafe helper to generate prompts with remaining words.
        Should only be called within a lock context.
        
        Args:
            base_prompt: Template prompt with placeholders
            noun_words: List of nouns to include in prompt
            
        Returns:
            str: Complete prompt ready to send to model
        """
        async_unsafe_adjs = list(self.expected_adjs)
        return prepare_prompt(base_prompt, self.all_sentences, noun_words, async_unsafe_adjs)

# ================================================================================
# === Helper Functions - Utilities for prompt preparation and processing =========
# ================================================================================
def prepare_prompt(base_prompt: str, current_sentences: List[dict],
                   noun_words: List[str], adj_words: List[str]) -> str:
    """
    Prepare a prompt by filling in placeholders with word lists.
    Orders adjectives based on frequency of existing sentences.
    
    Args:
        base_prompt: Template prompt with placeholders
        current_sentences: Currently collected sentences
        noun_words: List of nouns to include
        adj_words: List of adjectives to include
        
    Returns:
        str: Complete prompt ready to send to model
    """
    # Compute adjective order prioritizing those with fewer sentences
    if current_sentences:
        adj_counter = Counter(entry["word"] for entry in current_sentences if entry["word"] in adj_words)
        ordered_adjs = sorted(adj_words, key=lambda w: adj_counter.get(w, 0))
    else:
        ordered_adjs = adj_words
    adj_order_str = ", ".join(ordered_adjs)
    noun_order_str = ", ".join(noun_words)
    
    # Replace placeholders in template
    modified_prompt = base_prompt.replace("{adjective_order}", adj_order_str)
    modified_prompt = modified_prompt.replace("{noun_order}", noun_order_str)
    return modified_prompt


# ================================================================================
# === Async Ollama Client Manager - Resource management for API client ===========
# ================================================================================
@asynccontextmanager
async def get_ollama_client():
    """
    Async context manager for Ollama client to ensure proper cleanup.
    
    Yields:
        ollama.AsyncClient: Initialized async client for Ollama API
    """
    client = ollama.AsyncClient()
    try:
        yield client
    finally:
        # Ensure client resources are cleaned up properly
        pass

# ================================================================================
# === Async Model Execution - Functions for calling models and parsing results ===
# ================================================================================
async def run_model_async(client, model: str, prompt: str, temp: float, seed: int, 
                         run_id: int) -> Tuple[Optional[List[Dict[str, Any]]], List[str]]:
    """
    Executes the model call asynchronously and retrieves structured output.
    
    Args:
        client: Ollama AsyncClient instance
        model: Name of the model to use
        prompt: Prompt to send to the model
        temp: Temperature setting for generation
        seed: Random seed for reproducibility
        run_id: Identifier for this specific run
        
    Returns:
        Tuple of:
          - structured_output: list of sentences (as dictionaries)
          - errors: list of error messages, if any
    """
    # Track execution time for performance monitoring
    start_time = time.time()
    logger.info(f"[Batch {run_id}] Running model '{model}' with temperature={temp} and seed={seed}")
    logger.debug(f"[Batch {run_id}] Prompt sent to model: {prompt[:200]}...")
    
    # Try to call the model with error handling
    try:
        # Gebruik asyncio.wait_for om een timeout toe te passen
        model_task = client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            format=OutputSchema.model_json_schema(),  # Enforced output schema
            options={"temperature": temp, "seed": seed}
        )
        
        # Wacht voor maximaal MODEL_TIMEOUT seconden
        response = await asyncio.wait_for(model_task, timeout=MODEL_TIMEOUT)
        
        elapsed = time.time() - start_time
        logger.debug(f"[Batch {run_id}] Model response time: {elapsed:.2f} seconds")
    except asyncio.TimeoutError:
        # Specifieke timeout handling
        error_msg = f"[Batch {run_id}] Model call timed out after {MODEL_TIMEOUT} seconds"
        logger.error(error_msg)
        return None, [error_msg]
    except Exception as e:
        error_msg = f"[Batch {run_id}] Model call failed for {model}, temperature={temp}, seed={seed}: {e}"
        logger.error(error_msg)
        return None, [str(e)]
    
    # Parse and validate response
    errors = []
    try:
        # Validate response against our schema
        data = OutputSchema.model_validate_json(response['message']['content'])
        structured_output = [entry.dict() for entry in data.sentences[:10]]
        
        # Check if we got exactly 10 sentences as expected
        if len(structured_output) != 10:
            err_msg = f"Expected exactly 10 sentences, but got {len(structured_output)}"
            errors.append(err_msg)
            logger.error(f"[Batch {run_id}] {err_msg}")
        else:
            logger.info(f"[Batch {run_id}] Successfully parsed 10 sentences from model '{model}'")
    except Exception as e:
        error_msg = f"[Batch {run_id}] JSON validation failed for {model}, temperature={temp}, seed={seed}: {e}"
        logger.error(error_msg)
        structured_output = []
        errors.append(str(e))
    
    return structured_output, errors

# ================================================================================
# === Batch Processing - Logic for batching requests to Ollama API ==============
# ================================================================================
async def run_batch(client, model: str, shared_state: SharedState, base_prompt: str, 
                   temp: float, original_nouns: List[str], batch_size: int) -> bool:
    """
    Run a batch of model calls in parallel, using 10 random words for each prompt.
    
    Args:
        client: Ollama AsyncClient instance
        model: Name of model to call
        shared_state: Thread-safe state for tracking results
        base_prompt: Template prompt text
        temp: Temperature parameter for generation
        original_nouns: List of nouns to use in prompts
        batch_size: Maximum number of parallel requests
        
    Returns:
        bool: True if we've reached the target sentence count
    """
    tasks = []
    prompt_id = 0

    # 🔒 Thread-safely get remaining words
    async with shared_state.lock:
        # Check if we've already reached our target
        if len(shared_state.all_sentences) >= TOTAL_TARGET_SENTENCES:
            return True
            
        # Get all available words (words that haven't hit the max count yet)
        remaining_adjs = [word for word in shared_state.expected_adjs 
                         if shared_state.word_counter[word] < TARGET_COUNT_PER_WORD]
        
        # If no words remain below the threshold, we're done
        if not remaining_adjs:
            return True

    # Create 5 batches (or fewer if we don't have enough batch capacity)
    actual_batch_size = min(batch_size, 5)  # Limit to 5 parallel prompts
    
    for _ in range(actual_batch_size):
        # Select 10 random words for each prompt
        # If we have fewer than 10 words left, use all of them
        if len(remaining_adjs) <= 10:
            selected_words = remaining_adjs.copy()
        else:
            selected_words = random.sample(remaining_adjs, 10)
        
        # Create unique seed for this run
        seed = SEED_START + shared_state.run_count
        
        # Create prompt with the selected words
        prompt = prepare_prompt(base_prompt, [], original_nouns, selected_words)
        
        logger.debug(f"[Batch {prompt_id}] Seed: {seed}, Using 10 random words: {selected_words}")
        
        # Schedule model call
        task = run_model_async(client, model, prompt, temp, seed, prompt_id)
        tasks.append(task)
        prompt_id += 1
        
        # Update tracking stats
        async with shared_state.lock:
            shared_state.seeds_used.add(seed)
            shared_state.run_count += 1

    # Run all batches in parallel and collect results
    results = await asyncio.gather(*tasks)
    
    # Update shared state with all results
    target_reached = False
    for structured_output, errors in results:
        batch_reached_target = await shared_state.update_with_sentences(structured_output, errors)
        if batch_reached_target:
            target_reached = True
            break  # Exit early if we've reached the target

    return target_reached

# ================================================================================
# === Main Processing Logic - Per-prompt aggregation function ===================
# ================================================================================
async def run_aggregation(prompt_file: str, target_count: int = TARGET_COUNT_PER_WORD, 
                        models=None, temperatures=None):
    """
    Process a single prompt file, testing all model and temperature combinations.
    
    Args:
        prompt_file: Path to the prompt template file
        target_count: Target number of sentences per word
        models: List of models to test (defaults to MODELS)
        temperatures: List of temperatures to test (defaults to TEMPERATURES)
    """
    global TARGET_COUNT_PER_WORD, TOTAL_TARGET_SENTENCES
    if models is None:
        models = MODELS
    
    if temperatures is None:
        temperatures = TEMPERATURES
    
    # Extract gender information from prompt file name for proper categorization
    prompt_filename = os.path.basename(prompt_file)
    file_without_ext = os.path.splitext(prompt_filename)[0]
    parts = file_without_ext.replace("prompt_", "").split("_")
    noun_group = parts[0]       # e.g. "maleNouns" or "femaleNouns"
    adjective_group = parts[1]  # e.g. "maleAdjs" or "femaleAdjs"
    noun_gender = "male" if noun_group.lower().startswith("male") else "female"
    adjective_gender = "male" if adjective_group.lower().startswith("male") else "female"

    logger.info(f"Processing file {prompt_filename}: noun_gender: {noun_gender}, adjective_gender: {adjective_gender}")

    # Set expected words based on gender
    original_nouns = male_nouns if noun_gender == "male" else female_nouns
    original_adjs = male_adjs if adjective_gender == "male" else female_adjs
    
    # Read the base prompt content
    try:
        with open(prompt_file, 'r', encoding='utf-8') as pf:
            base_prompt = pf.read()
        logger.info(f"Successfully read the prompt from {prompt_file}")
    except Exception as e:
        logger.exception(f"Failed to read the prompt file {prompt_file}: {e}")
        return
    
    TOTAL_TARGET_SENTENCES = 200
    
    # Process each model and temperature combination
    async with get_ollama_client() as client:
        for model in models:
            # Create a model-specific log directory for organized output
            model_dir = os.path.join("Notebooks/Phase_02/logs", model.replace(":", "_"))
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created log directory for model '{model}': {model_dir}")
            
            for temp in temperatures:
                # Initialize fresh shared state for each temperature setting
                shared_state = SharedState(
                    word_counter=Counter({word: 0 for word in original_adjs}),
                    expected_adjs=original_adjs[:],
                    expected_adjs_set=set(original_adjs),
                    run_count=0,
                    all_sentences=[],
                    aggregated_errors=[],
                    noun_gender=noun_gender,
                    adjective_gender=adjective_gender,
                    model_name=model,
                    temperature=temp,
                    last_save_time=time.time()
                )
                
                logger.info(f"Starting aggregation for model '{model}' with temperature {temp}")
                
                # Time tracking for overall process
                experiment_start_time = time.time()
                last_progress_count = 0
                consecutive_no_progress = 0
                
                while len(shared_state.all_sentences) < TOTAL_TARGET_SENTENCES and shared_state.expected_adjs:
                    # Check if experiment timed out
                    current_time = time.time()
                    experiment_elapsed = current_time - experiment_start_time
                    
                    # Calculate optimal batch size based on remaining work
                    async with shared_state.lock:
                        remaining_count = TOTAL_TARGET_SENTENCES - sum(shared_state.word_counter.values())
                        remaining_words = len(shared_state.expected_adjs_set)
                        sentences_collected = len(shared_state.all_sentences)
                        logger.info(f"Collected {sentences_collected} sentences after {experiment_elapsed:.1f} seconds")
                    
                    # Early termination if no progress in last few batches
                    if sentences_collected == last_progress_count:
                        consecutive_no_progress += 1
                        if consecutive_no_progress >= 5:  # No progress in 5 consecutive batches
                            logger.warning(f"No progress after {consecutive_no_progress} batches, terminating early")
                            break
                    else:
                        consecutive_no_progress = 0
                        last_progress_count = sentences_collected
                    
                    # Determine batch size: don't create more tasks than needed words
                    batch_size = min(MAX_BATCH_SIZE, remaining_words)
                    if batch_size == 0:
                        logger.info("No more sentences needed, exiting collection loop")
                        break
                    
                    logger.info(f"Running batch of {batch_size} prompts for model '{model}' with temperature {temp}")
                    
                    # Run the batch and check if all targets are met
                    all_met = await run_batch(
                        client, model, shared_state, base_prompt,
                        temp, original_nouns, batch_size
                    )
                    
                    if all_met:
                        logger.info("All expected words have reached the target count.")
                        break
                
                # Safely access final state to prepare output
                async with shared_state.lock:
                    # Save final intermediate results
                    await shared_state.save_intermediate_results()
                    
                    # Trim to exactly the TOTAL_TARGET_SENTENCES if we have more
                    if len(shared_state.all_sentences) > TOTAL_TARGET_SENTENCES:
                        logger.info(f"Trimming output from {len(shared_state.all_sentences)} to {TOTAL_TARGET_SENTENCES} sentences")
                        shared_state.all_sentences = shared_state.all_sentences[:TOTAL_TARGET_SENTENCES]
                
                    # Create a log filename with prompt and temperature information
                    log_filename = f"{noun_group}-{adjective_group}_temp{temp}.jsonl"
                    log_file_path = os.path.join(model_dir, log_filename)
                  
                    # Build aggregated log data with experiment metadata
                    log_data = {
                        "model": model,
                        "temperature": temp,
                        "total_runs": shared_state.run_count,
                        "seeds_used": list(shared_state.seeds_used),
                        "aggregated_output": shared_state.all_sentences,
                        "total_sentences": len(shared_state.all_sentences),
                        "parse_errors": shared_state.aggregated_errors,
                        "noun_gender": noun_gender,
                        "adjective_gender": adjective_gender,
                        "final_word_counts": dict(shared_state.word_counter),
                        "total_runtime_seconds": time.time() - experiment_start_time
                    }
                
                try:
                    # Write log data atomically (important for async processes)
                    with open(log_file_path, "w", encoding="utf-8") as lf:
                        json.dump(log_data, lf, ensure_ascii=False, indent=2)
                    logger.info(f"Logged aggregated output to {log_file_path} for model '{model}' with temperature {temp}")
                except Exception as e:
                    logger.exception(f"Failed to write log file {log_file_path}: {e}")

# ================================================================================
# === Main Entry Point - Script coordination function ===========================
# ================================================================================
async def main():
    """
    Main entry point for the script.
    Finds all prompt files and processes each one for all model/temperature combinations.
    """
    logger.info("===== EXPERIMENT START =====")
    logger.info(f"Processing with targets: {TOTAL_TARGET_SENTENCES} total sentences, max {TARGET_COUNT_PER_WORD} per word")
    logger.info(f"Using models: {MODELS} with temperatures: {TEMPERATURES}")
    
    # Find all prompt files
    prompt_files = glob.glob(os.path.join(PROMPT_FOLDER, "*.txt"))
    print(f"Found prompt files: {prompt_files}")
    
    if not prompt_files:
        err_msg = f"No prompt files found in {PROMPT_FOLDER}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    
    logger.info(f"Found {len(prompt_files)} prompt files: {[os.path.basename(f) for f in prompt_files]}")
    
    # Process each prompt file sequentially
    for prompt_file in prompt_files:
        logger.info(f"===== PROCESSING {os.path.basename(prompt_file)} =====")
        await run_aggregation(prompt_file)
    
    logger.info("===== EXPERIMENT COMPLETE =====")
    logger.info("Completed aggregating outputs for all models and temperatures.")

# ================================================================================
# === Script Execution - Entrypoint with error handling =========================
# ================================================================================
if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle clean shutdown on Ctrl+C
        logger.info("Process interrupted by user. Shutting down gracefully.")
    except Exception as e:
        # Log any unhandled exceptions
        logger.exception(f"Unhandled exception in main process: {e}")
        raise  # Re-raise to show error in console
    
    
    ## Obscuring bias by methodology. 
    ## Qualitative discussion
    ## Proportion of dutch text the model were trained on; what to expect of this mulitlingual models and these language
    ## Way of work: Reproduction study between Langugges, dont dwell on it to much, probably mention as a deficit: reproduction study, hard for NLP