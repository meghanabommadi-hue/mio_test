"""Sample sentences used by benchmark.py and ws_loadtest.py.

Kept dependency-free (no torch/soundfile/etc.) so it can be imported from
either venv -- including .venv_vllm, which deliberately doesn't have the
audio/codec stack installed.
"""

SAMPLE_TEXTS = {
    "generic": {
        "english": [
            "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Welcome to the demonstration of the text to speech system.",
            "This model supports many Indian languages and English.",
            "Artificial intelligence is transforming how we communicate.",
        ],
        "hindi": [
            "नमस्ते, आप कैसे हैं?",
            "आज मौसम बहुत अच्छा है।",
            "भारत एक विविधतापूर्ण देश है।",
            "यह एक पाठ से वाक् प्रणाली का परीक्षण है।",
            "मुझे हिंदी में बात करना पसंद है।",
        ],
        "telugu": [
            "నమస్కారం, మీరు ఎలా ఉన్నారు?",
            "ఈ రోజు వాతావరణం చాలా బాగుంది.",
            "తెలుగు ఒక అందమైన భాష.",
            "ఇది టెక్స్ట్ టు స్పీచ్ వ్యవస్థ యొక్క పరీక్ష.",
            "నాకు తెలుగులో మాట్లాడటం ఇష్టం.",
        ],
    },
    "collections": {
        "english": [
            "Hi, this is Vaani from Bajaj Finance.",
            "This call is on a recorded line.",
            "It's about your loan account ending five six two eight.",
            "Your E M I bounced this month.",
            "Overdue amount is six thousand five hundred rupees.",
            "Can you make the payment right now?",
            "You can pay through the Bajaj Finance App.",
            "Do you already have the Bajaj App installed?",
        ],
        "hindi": [
            "नमस्ते, मैं Bajaj Finance से वाणी बोल रही हूं।",
            "यह कॉल एक recorded line पर है।",
            "यह आपके loan account के बारे में है, जो पांच छह दो आठ पर खत्म होता है।",
            "इस महीने आपका E M I bounce हो गया।",
            "Overdue amount छह हज़ार पाँच सौ रुपये है।",
            "क्या आप अभी payment कर सकते हैं?",
            "आप Bajaj Finance App से भी payment कर सकते हैं।",
            "क्या आपके फ़ोन में Bajaj App installed है?",
        ],
        "telugu": [
            "నమస్తే, నేను Bajaj Finance నుండి Vaani ని.",
            "ఈ call ఒక recorded line మీద జరుగుతోంది.",
            "ఇది మీ loan account గురించి, ఐదు ఆరు రెండు ఎనిమిది తో end అవుతుంది.",
            "ఈ నెల మీ E M I bounce అయింది.",
            "Overdue amount ఆరు వేల ఐదు వందల రూపాయలు.",
            "మీరు ఇప్పుడే payment చేస్తారా?",
            "మీరు Bajaj Finance App తో కూడా payment చేయవచ్చు.",
            "మీ ఫోన్‌లో Bajaj App installed ఉందా?",
        ],
        "punjabi": [
            "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ Bajaj Finance ਤੋਂ ਵਾਣੀ।",
            "ਇਹ ਕਾਲ ਇੱਕ recorded line ਤੇ ਹੈ।",
            "ਇਹ ਤੁਹਾਡੇ loan account ਬਾਰੇ ਹੈ, ਜੋ ਪੰਜ ਛੇ ਦੋ ਅੱਠ ਤੇ ਖਤਮ ਹੁੰਦਾ ਹੈ।",
            "ਇਸ ਮਹੀਨੇ ਤੁਹਾਡਾ E M I bounce ਹੋ ਗਿਆ।",
            "Overdue amount ਛੇ ਹਜ਼ਾਰ ਪੰਜ ਸੌ ਰੁਪਏ ਹੈ।",
            "ਕੀ ਤੁਸੀਂ ਹੁਣੇ payment ਕਰ ਸਕਦੇ ਹੋ?",
            "ਤੁਸੀਂ Bajaj Finance App ਨਾਲ ਵੀ payment ਕਰ ਸਕਦੇ ਹੋ।",
            "ਕੀ ਤੁਹਾਡੇ ਫ਼ੋਨ ਵਿੱਚ Bajaj App installed ਹੈ?",
        ],
        "marathi": [
            "नमस्कार, मी Bajaj Finance कडून वाणी बोलते आहे.",
            "हा कॉल एका recorded line वर आहे.",
            "हे तुमच्या loan account बद्दल आहे, जे पाच सहा दोन आठ ने संपते.",
            "या महिन्यात तुमचा E M I bounce झाला.",
            "Overdue amount सहा हजार पाचशे रुपये आहे.",
            "तुम्ही आत्ताच payment करू शकता का?",
            "तुम्ही Bajaj Finance App ने पण payment करू शकता.",
            "तुमच्या फोनमध्ये Bajaj App installed आहे का?",
        ],
        "assamese": [
            "নমস্কাৰ, মই Bajaj Finance ৰ পৰা ভাণী।",
            "এই call টো এখন recorded line ত হৈ আছে।",
            "এইটো আপোনাৰ loan account ৰ বিষয়ে, যিটো পাঁচ ছয় দুই আঠত শেষ হয়।",
            "এই মাহত আপোনাৰ E M I bounce হৈ গল।",
            "Overdue amount ছয় হাজাৰ পাঁচ শ টকা।",
            "আপুনি এতিয়াই payment কৰিব পাৰেনে?",
            "আপুনি Bajaj Finance App ৰেও payment কৰিব পাৰে।",
            "আপোনাৰ ফোনত Bajaj App installed আছে নে?",
        ],
        "gujarati": [
            "નમસ્તે, હું Bajaj Finance માંથી વાણી.",
            "આ call એક recorded line પર છે.",
            "આ તમારા loan account વિશે છે, જે પાંચ છ બે આઠ પર પૂરો થાય છે.",
            "આ મહિને તમારું E M I bounce થયું.",
            "Overdue amount છ હજાર પાંચસો રૂપિયા છે.",
            "શું તમે અત્યારે payment કરી શકો છો?",
            "તમે Bajaj Finance App થી પણ payment કરી શકો છો.",
            "શું તમારા ફોનમાં Bajaj App installed છે?",
        ],
    },
}
