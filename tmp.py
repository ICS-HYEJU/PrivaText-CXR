from transformers import AutoModel

# Hugging Face¿¡¼­ BioViL-T ¸ðµ¨ ·Îµå
model = AutoModel.from_pretrained(
    "microsoft/BiomedVLP-BioViL-T",
    trust_remote_code=True,
    dtype="auto"
)
print(model)