from transformers import CLIPTextModel, CLIPTokenizer


# 文本编码
def prompts_embedding(prompts):
    # 加载编码模型
    tokenizer = CLIPTokenizer.from_pretrained("/home/duomeitinrfx/users/pengxl/multi label class/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("/home/duomeitinrfx/users/pengxl/multi label class/clip-vit-large-patch14")

    # tokenizer.model_max_length -> 77
    text_input = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                           return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids)
    text_embeddings = text_embeddings[0]  # (1, 77, 768)

    return text_embeddings


def embedding():
    prompts = ["测试文本"]
    text_embeddings = prompts_embedding(prompts)

    uncond_prompts = [""]
    uncond_embeddings = prompts_embedding(uncond_prompts)
    print(uncond_embeddings)

    print("text_embeddings.shape", text_embeddings.shape)
    print("text_embeddings.shape", uncond_embeddings.shape)


embedding()