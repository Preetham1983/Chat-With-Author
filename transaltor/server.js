const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 8000;

app.use(cors());
app.use(express.json());

app.post('/translate', async (req, res) => {
    const { text, to } = req.body;

    try {
        const response = await axios.get('https://translate.googleapis.com/translate_a/single', {
            params: {
                client: 'gtx',
                sl: 'auto',
                tl: to,
                dt: 't',
                q: text
            }
        });
        let translatedText = '';
        if (response.data && response.data[0]) {
            response.data[0].forEach((sentence) => {
                if (sentence[0]) {
                    translatedText += sentence[0];
                }
            });
        }

        res.json({ translatedText });
    } catch (error) {
        console.error('Error translating text:', error);
        res.status(500).json({ error: 'Translation error' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

