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
    "corrupt", "onoverwinnelijk", "plaatsvervangend", "impopulair", "goddeloos",
    "incompetent", "misdadig", "bekwaam", "sadistisch", "gewetenloos", 
    "steenrijk", "vooraanstaand", "voortvluchtig", "geniaal", "planmatig"
]

# Dutch adjectives typically associated with females in the experiment
female_adjs = [
    "blond", "beeldschoon", "bloedmooie", "donkerharig", "ongehuwd",
    "kinderloos", "glamoureus", "beeldig", "sensueel", "platinablond", 
    "voorlijk", "feministisch", "stijlvol", "tuttig", "rimpelig"
]

# ================================================================================
# === Experiment Configuration - Adjustable parameters for the experiment ========
# ================================================================================
# Directory containing prompt template files
PROMPT_FOLDER = "Notebooks/Phase_02/prompts"
# Models to test - we're using local Ollama models
MODELS = ["llama3:text"]
# Temperature values to test for diversity in responses
TEMPERATURES = [1]
# Base seed for reproducibility
SEED_START = 42
# Parallel processing capacity - controls simultaneous API calls
MAX_BATCH_SIZE = 5
# Number of sentences to collect per adjective
TARGET_COUNT_PER_WORD = 15

# ================================================================================
# === Shared State Manager - Thread-safe data store for experiment results =======
# ================================================================================
@dataclass
class SharedState:
    """
    Thread-safe state manager for processing and storing experiment results.
    Uses asyncio.Lock for thread safety in the async environment.
    """
    # Count of sentences collected per adjective
    word_counter: Counter = field(default_factory=Counter)
    # Complete collection of generated sentences
    all_sentences: List[Dict[str, Any]] = field(default_factory=list)
    # List of adjectives to collect sentences for
    expected_adjs: List[str] = field(default_factory=list)
    # Set for faster membership checking
    expected_adjs_set: Set[str] = field(default_factory=set)
    # Count of model executions
    run_count: int = 0
    # Collection of errors encountered
    aggregated_errors: List[str] = field(default_factory=list)
    # Async mutex for thread safety
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Track seeds used for reproducibility analysis
    seeds_used: Set[int] = field(default_factory=set)

    async def update_with_sentences(self, structured_output: List[Dict[str, Any]], errors: List[str]) -> bool:
        """
        Thread-safe method to update the shared state with new sentences and track errors.
        
        Args:
            structured_output: List of sentence entries from a model response
            errors: List of error messages encountered during processing
            
        Returns:
            bool: True if all target words have met their quota, False otherwise
        """
        async with self.lock:
            # Log any errors encountered
            if errors:
                self.aggregated_errors.extend(errors)
                logger.warning(f"Encountered errors: {errors}")

            all_words_met = True
            added_count = 0
            
            if structured_output:
                for entry in structured_output:
                    word = entry["word"]
                    sentence = entry.get("sentence", "")
                    
                    # Skip words not in our target list
                    if word not in self.expected_adjs_set:
                        logger.debug(f"Skipping unexpected word '{word}' from output.")
                        continue
                        
                    # Add sentence if we haven't reached target count for this word
                    if self.word_counter[word] < TARGET_COUNT_PER_WORD:
                        self.all_sentences.append(entry)
                        self.word_counter[word] += 1
                        added_count += 1
                        
                        # If target reached, remove word from pending list
                        if self.word_counter[word] >= TARGET_COUNT_PER_WORD:
                            try:
                                self.expected_adjs.remove(word)
                                self.expected_adjs_set.remove(word)
                                logger.info(f"Word '{word}' reached target ({TARGET_COUNT_PER_WORD}) and was removed from adjective list.")
                            except ValueError:
                                logger.warning(f"Tried to remove word '{word}', but it was already removed.")
                    else:
                        logger.debug(f"Skipping sentence with word '{word}' because its count reached the target.")

            # Check if all words have met their quota
            for word in list(self.expected_adjs_set):
                if self.word_counter[word] < TARGET_COUNT_PER_WORD:
                    all_words_met = False

            # Log progress if new sentences were added
            if added_count > 0:
                logger.info(f"Added {added_count} new sentences; Total sentences: {len(self.all_sentences)}")
                logger.info(f"Current word counts: {dict(self.word_counter)}")
                logger.info(f"Words still needing sentences: {sorted(self.expected_adjs_set)}")
            
            return all_words_met

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
        await client.close()

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
        response = await client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            format=OutputSchema.model_json_schema(),  # Enforced output schema
            options={"temperature": temp, "seed": seed}
        )
        elapsed = time.time() - start_time
        logger.debug(f"[Batch {run_id}] Model response time: {elapsed:.2f} seconds")
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
    Run a batch of model calls in parallel, optimizing for remaining words.
    
    Args:
        client: Ollama AsyncClient instance
        model: Name of model to call
        shared_state: Thread-safe state for tracking results
        base_prompt: Template prompt text
        temp: Temperature parameter for generation
        original_nouns: List of nouns to use in prompts
        batch_size: Maximum number of parallel requests
        
    Returns:
        bool: True if all target words have met their quota
    """
    tasks = []
    prompt_id = 0

    # 🔒 Thread-safely get remaining words
    async with shared_state.lock:
        usage_map = dict(shared_state.word_counter)
        remaining_adjs = [word for word in shared_state.expected_adjs if shared_state.word_counter[word] < TARGET_COUNT_PER_WORD]
        remaining_count = len(remaining_adjs)

    # Adapt batch strategy based on remaining words
    if remaining_count <= batch_size:
        # Create one batch per word when we have fewer words than batch_size
        # This maximizes efficiency in final stages
        selected_batches = [[word] for word in remaining_adjs]
        logger.debug(f"Few words remaining ({remaining_count}), creating single-word batches")
    else:
        # ➕ Group words by how many sentences we already have for each
        usage_groups = {}
        for word in remaining_adjs:
            count = usage_map[word]
            usage_groups.setdefault(count, []).append(word)

        # 🧱 Step 1: Create single list prioritizing words with fewer sentences
        all_remaining_words = []
        for usage_count in sorted(usage_groups):
            all_remaining_words.extend(usage_groups[usage_count])

        # 🎲 Step 2: Shuffle this list to prevent getting stuck on certain words
        random.shuffle(all_remaining_words)

        # 📦 Step 3: Create batches of varying sizes (3, 2, or 1)
        batches = []
        i = 0
        while i < len(all_remaining_words):
            if i + 3 <= len(all_remaining_words):
                batch = all_remaining_words[i:i + 3]
                i += 3
            elif i + 2 <= len(all_remaining_words):
                batch = all_remaining_words[i:i + 2]
                i += 2
            else:
                batch = all_remaining_words[i:i + 1]
                i += 1
            batches.append(batch)

        # 🔀 Step 4: Shuffle batches for variation in prompts
        random.shuffle(batches)

        # ✂️ Step 5: Limit to requested batch size
        selected_batches = batches[:batch_size]
        logger.debug(f"Created {len(selected_batches)} batches from {remaining_count} remaining words")

    # 🚀 Step 6: Start model tasks for each batch
    for batch_words in selected_batches:
        # Additional shuffle within batch for variation
        random.shuffle(batch_words)

        # Create unique seed for this run
        seed = SEED_START + shared_state.run_count
        prompt = prepare_prompt(base_prompt, [], original_nouns, batch_words)

        logger.debug(f"[Batch {prompt_id}] Seed: {seed}, Group size: {len(batch_words)}, Words: {batch_words}")

        # Schedule model call
        task = run_model_async(client, model, prompt, temp, seed, prompt_id)
        tasks.append(task)
        prompt_id += 1

        # Update tracking stats
        async with shared_state.lock:
            shared_state.seeds_used.add(seed)
            shared_state.run_count += 1

    # ✅ Run all batches in parallel and collect results
    results = await asyncio.gather(*tasks)

    # 🧠 Update shared state with all results
    all_words_met = False
    for structured_output, errors in results:
        words_met = await shared_state.update_with_sentences(structured_output, errors)
        if words_met:
            all_words_met = True

    return all_words_met

# ================================================================================
# === Main Processing Logic - Per-prompt aggregation function ===================
# ================================================================================
async def run_aggregation(prompt_file: str):
    """
    Process a single prompt file, testing all model and temperature combinations.
    
    Args:
        prompt_file: Path to the prompt template file
    """
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
    
    TOTAL_TARGET_SENTENCES = TARGET_COUNT_PER_WORD * len(original_adjs)
    logger.info(f"Target: {TOTAL_TARGET_SENTENCES} sentences ({TARGET_COUNT_PER_WORD} per word × {len(original_adjs)} words)")
    
    # Process each model and temperature combination
    async with get_ollama_client() as client:
        for model in MODELS:
            # Create a model-specific log directory for organized output
            model_dir = os.path.join("Notebooks/Phase_02/logs", model.replace(":", "_"))
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created log directory for model '{model}': {model_dir}")
            
            for temp in TEMPERATURES:
                # Initialize fresh shared state for each temperature setting
                shared_state = SharedState(
                    word_counter=Counter({word: 0 for word in original_adjs}),
                    expected_adjs=original_adjs[:],
                    expected_adjs_set=set(original_adjs),
                    run_count=0,
                    all_sentences=[],
                    aggregated_errors=[]
                )
                
                logger.info(f"Starting aggregation for model '{model}' with temperature {temp}")
                
                # Continue until all expected words have reached the target count
                while sum(shared_state.word_counter.values()) < TOTAL_TARGET_SENTENCES:
                    # Calculate optimal batch size based on remaining work
                    async with shared_state.lock:
                        remaining_count = TOTAL_TARGET_SENTENCES - sum(shared_state.word_counter.values())
                        remaining_words = len(shared_state.expected_adjs_set)
                        logger.info(f"Remaining words: {remaining_words}, remaining sentences needed: {remaining_count}")
                    
                    # Determine batch size: don't create more tasks than needed words
                    batch_size = min(MAX_BATCH_SIZE, remaining_count)
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
                    # Trim to exactly the TOTAL_TARGET_SENTENCES if we have more
                    if len(shared_state.all_sentences) > TOTAL_TARGET_SENTENCES:
                        logger.info(f"Trimming output from {len(shared_state.all_sentences)} to {TOTAL_TARGET_SENTENCES} sentences")
                        shared_state.all_sentences = shared_state.all_sentences[:TOTAL_TARGET_SENTENCES]
                
                    # Create a log filename with prompt and temperature information
                    log_filename = f"{noun_group}-{adjective_group}_temp{temp}_0_5.jsonl"
                    log_file_path = os.path.join(model_dir, log_filename)
                  
                    # Build aggregated log data with experiment metadata
                    log_data = {
                        "model": model,
                        "temperature": temp,
                        "total_runs": shared_state.run_count,
                        "seeds_used": list(range(SEED_START, SEED_START + shared_state.run_count)),
                        "aggregated_output": shared_state.all_sentences,
                        "total_sentences": len(shared_state.all_sentences),
                        "parse_errors": shared_state.aggregated_errors,
                        "noun_gender": noun_gender,
                        "adjective_gender": adjective_gender,
                        "final_word_counts": dict(shared_state.word_counter)
                    }
                
                try:
                    # Write log data atomically (important for async processes)
                    with open(log_file_path, "a", encoding="utf-8") as lf:
                        json.dump(log_data, lf, ensure_ascii=False)
                        lf.write("\n")
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
    logger.info(f"Processing with targets: {TARGET_COUNT_PER_WORD} sentences per word")
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