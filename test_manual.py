
from main import convert_banglish_to_bangla

test_sentences = [
    "ami valo achi",
    "tumi kemon acho",
    "ami banglay kotha bolte chai",
    "amar bangla lekha shikhte hobe",
    "ajke abohawa onek valo",
    "ami university te porashona kori",
    "tumi ki amar kotha bujhte parcho",
    "ami ekhon bari jabo",
    "amar bondhu amar sathe dekha korbe",
    "ami machine learning shikhchi",
]

for sentence in test_sentences:
    prediction = convert_banglish_to_bangla(sentence)

    print(f"\nBanglish: {sentence}")
    print(f"Prediction: {prediction}")
