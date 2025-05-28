from flask import Flask, request, jsonify
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

app = Flask(__name__)

model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

lang_code = {"English": "en", "Chinese": "zh", "Arabic": "ar"}

def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = lang_code[source_lang]
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(lang_code[target_lang])
    )
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data["text"]
    src = data["source_lang"]
    tgt = data["target_lang"]
    translated = translate_text(text, src, tgt)
    return jsonify({"translated_text": translated})

if __name__ == "__main__":
    app.run(debug=True)
