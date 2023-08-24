#%%
import transformers as t
import torch
import peft
import time
import wandb
from wandb.sdk import data_types
#%%
tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
model = t.AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", load_in_8bit=True, torch_dtype=torch.float16)
tokenizer.pad_token_id = 0
#%%
config = peft.LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.005, bias="none", task_type="CAUSAL_LM")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = peft.get_peft_model(model, config).to(device)

wandb.init(
        project="Llama",
        name = "Llama fine-tune eval tests",
        config={
})

table = wandb.Table(columns=["Title", "Generated Article", "Attention"])

TEMPLATE ="Below is a title for an article. Write an article that appropriately suits the title: \n\n### Title:\n{title}\n\n### Article:\n"
elm = {"Title": "Machine learning students brutally killed after annoying Bes"}
prompt = TEMPLATE.format(title=elm["Title"])

for checkpoint in [1200, 1600]:
    peft.set_peft_model_state_dict(model, torch.load(f"./output/checkpoint-{checkpoint}/adapter_model.bin"))
    # peft.set_peft_model_state_dict(model, torch.load("./output/checkpoint-1400/adapter_model.bin"))

    # %% 
    pipe = t.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000)
    output = pipe([prompt])
    # output = pipe(prompt)
    print("pipe(prompt)", output)
    generated_article = output[0][0]["generated_text"]
    # attention = output[1]
    attention = "none"

    table.add_data(elm["Title"], generated_article, attention)