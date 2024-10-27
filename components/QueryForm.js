// import React, { useState } from 'react';
// import axios from 'axios';
// import './QueryForm.css';

// const QueryForm = () => {
//   const [userQuestion, setUserQuestion] = useState('');
//   const [response, setResponse] = useState('');
//   const [translatedResponse, setTranslatedResponse] = useState('');
//   const [error, setError] = useState('');
//   const [language, setLanguage] = useState('en'); // Default to English

//   const handleSubmit = async (e) => {
//     e.preventDefault(); // Prevent default form submission behavior

//     try {
//       const res = await axios.post('http://127.0.0.1:5000/process-query', {
//         user_question: userQuestion
//       });

//       const generatedResponse = res.data.generated_response;
//       if (generatedResponse === "I'm sorry, I cannot answer this question based on the provided context.") {
//         setError('');
//       }
//       setResponse(generatedResponse);
//       setTranslatedResponse(generatedResponse); // Initialize translated response with the original response
//     } catch (error) {
//       console.error('Error processing query:', error);
//       setError('Error: Unable to get response');
//     }
//   };

//   const handleLanguageChange = async (e) => {
//     const newLanguage = e.target.value;
//     setLanguage(newLanguage);

//     if (response) {
//       try {
//         const res = await axios.post('http://127.0.0.1:8000/translate', {
//           text: response,
//           to: newLanguage
//         });
//         setTranslatedResponse(res.data.translatedText);
//       } catch (error) {
//         console.error('Error translating response:', error);
//         setError('Error: Unable to translate response');
//       }
//     }
//   };

//   const handleTextToSpeech = () => {
//     const utterance = new SpeechSynthesisUtterance(translatedResponse);
//     utterance.lang = language;

//     // Find a suitable voice for the selected language
//     const voices = window.speechSynthesis.getVoices();
//     const voice = voices.find((v) => v.lang.startsWith(language));

//     if (voice) {
//       utterance.voice = voice;
//     } else {
//       // Handle the case where no suitable voice is found
//       console.error(`No voice found for language: ${language}`);
//       setError(`No voice found for language: ${language}`);
//     }

//     window.speechSynthesis.speak(utterance);
//   };

//   return (
//     <div className="query-form-container">
//       <form className="query-form" onSubmit={handleSubmit}>
//         <input
//           type="text"
//           value={userQuestion}
//           onChange={(e) => setUserQuestion(e.target.value)}
//           placeholder="Enter your question..."
//         />
//         <button type="submit">Ask Question</button>
//       </form>
//       {error && <div className="error-message">{error}</div>}
//       {response && (
//         <div>
//           <div className="language-selection">
//             <select value={language} onChange={handleLanguageChange}>
//               <option value="en">English</option>
//               <option value="hi">Hindi</option>
//               <option value="te">Telugu</option>
//               <option value="fr">French</option>
//               <option value="es">Spanish</option>
//               <option value="de">German</option>
//               <option value="it">Italian</option>
//               <option value="ja">Japanese</option>
//               <option value="ko">Korean</option>
//               <option value="zh-CN">Chinese (Simplified)</option>
//               <option value="zh-TW">Chinese (Traditional)</option>
//               <option value="ru">Russian</option>
//               <option value="ar">Arabic</option>
//               <option value="pt">Portuguese</option>
//               <option value="nl">Dutch</option>
//               <option value="sv">Swedish</option>
//               <option value="pl">Polish</option>
//               <option value="tr">Turkish</option>
//               <option value="vi">Vietnamese</option>
//               <option value="th">Thai</option>
//               <option value="el">Greek</option>
//             </select>
//             <button onClick={handleTextToSpeech}>Speak</button>
//           </div>

//           <div className="query-response">

//             {translatedResponse}
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default QueryForm;

// import React, { useState } from 'react';
// import axios from 'axios';
// import './QueryForm.css';

// const QueryForm = () => {
//   const [userQuestion, setUserQuestion] = useState('');
//   const [response, setResponse] = useState('');
//   const [translatedResponse, setTranslatedResponse] = useState('');
//   const [error, setError] = useState('');
//   const [language, setLanguage] = useState('en'); // Default to English
//   const [loading, setLoading] = useState(false); // New state for loading

//   // const handleSubmit = async (e) => {
//   //   e.preventDefault(); // Prevent default form submission behavior

//   //   setLoading(true); // Set loading state to true
//   //   setError(''); // Clear previous errors
//   //   setResponse(''); // Clear previous response

//   //   try {
//   //     const res = await axios.post('http://127.0.0.1:5000/process-query', {
//   //       user_question: userQuestion
//   //     });

//   //     const generatedResponse = res.data.generated_response;
//   //     if (generatedResponse === "I'm sorry, I cannot answer this question based on the provided context.") {
//   //       setError('');
//   //     }
//   //     setResponse(generatedResponse);
//   //     setTranslatedResponse(generatedResponse); // Initialize translated response with the original response
//   //   } catch (error) {
//   //     console.error('Error processing query:', error);
//   //     setError('Error: Unable to get response');
//   //   }

//   //   setLoading(false); // Set loading state to false
//   // };
//   const handleSubmit = async (e) => {
//     e.preventDefault(); // Prevent default form submission behavior

//     setLoading(true); // Set loading state to true
//     setError(''); // Clear previous errors
//     setResponse(''); // Clear previous response

//     try {
//       const res = await axios.post('http://127.0.0.1:5000/process-query', {
//         user_question: userQuestion
//       });

//       const generatedResponse = res.data.generated_response;
//       if (generatedResponse === "I'm sorry, I cannot answer this question based on the provided context.") {
//         setError('');
//       }
//       setResponse(generatedResponse);
//       setTranslatedResponse(generatedResponse); // Initialize translated response with the original response

//       // Save query and response to MongoDB
//       await axios.post('http://localhost:4000/save-query', {
//         userQuestion,
//         response: generatedResponse
//       });
//     } catch (error) {
//       console.error('Error processing query:', error);
//       setError('Error: Unable to get response');
//     }

//     setLoading(false); // Set loading state to false
//   };


//   const handleLanguageChange = async (e) => {
//     const newLanguage = e.target.value;
//     setLanguage(newLanguage);

//     if (response) {
//       try {
//         const res = await axios.post('http://127.0.0.1:8000/translate', {
//           text: response,
//           to: newLanguage
//         });
//         setTranslatedResponse(res.data.translatedText);
//       } catch (error) {
//         console.error('Error translating response:', error);
//         setError('Error: Unable to translate response');
//       }
//     }
//   };

//   const handleTextToSpeech = () => {
//     const utterance = new SpeechSynthesisUtterance(translatedResponse);
//     utterance.lang = language;

//     // Find a suitable voice for the selected language
//     const voices = window.speechSynthesis.getVoices();
//     const voice = voices.find((v) => v.lang.startsWith(language));

//     if (voice) {
//       utterance.voice = voice;
//     } else {
//       // Handle the case where no suitable voice is found
//       console.error(`No voice found for language: ${language}`);
//       setError(`No voice found for language: ${language}`);
//     }

//     window.speechSynthesis.speak(utterance);
//   };

//   return (
//     <div className="query-form-container">
//       <form className="query-form" onSubmit={handleSubmit}>
//         <input
//           type="text"
//           value={userQuestion}
//           onChange={(e) => setUserQuestion(e.target.value)}
//           placeholder="Enter your question..."
//         />
//         <button type="submit">Ask Question</button>
//       </form>
//       {error && <div className="error-message">{error}</div>}
//       {loading ? (
//         <div className="processing-message">Processing...</div>
//       ) : (
//         response && (
//           <div>
//             <div className="language-selection">
//               <select value={language} onChange={handleLanguageChange}>
//                 <option value="en">English</option>
//                 <option value="hi">Hindi</option>
//                 <option value="te">Telugu</option>
//                 <option value="fr">French</option>
//                 <option value="es">Spanish</option>
//                 <option value="de">German</option>
//                 <option value="it">Italian</option>
//                 <option value="ja">Japanese</option>
//                 <option value="ko">Korean</option>
//                 <option value="zh-CN">Chinese (Simplified)</option>
//                 <option value="zh-TW">Chinese (Traditional)</option>
//                 <option value="ru">Russian</option>
//                 <option value="ar">Arabic</option>
//                 <option value="pt">Portuguese</option>
//                 <option value="nl">Dutch</option>
//                 <option value="sv">Swedish</option>
//                 <option value="pl">Polish</option>
//                 <option value="tr">Turkish</option>
//                 <option value="vi">Vietnamese</option>
//                 <option value="th">Thai</option>
//                 <option value="el">Greek</option>
//               </select>
//               <button onClick={handleTextToSpeech}>Speak</button>
//             </div>
//             <div className="query-response">
//               {translatedResponse}
//             </div>
//           </div>
//         )
//       )}
//     </div>
//   );
// };

// export default QueryForm;

// import React, { useState } from 'react';
// import axios from 'axios';
// import './QueryForm.css';

// const QueryForm = () => {
//   const [userQuestion, setUserQuestion] = useState('');
//   const [response, setResponse] = useState('');
//   const [translatedResponse, setTranslatedResponse] = useState('');
//   const [error, setError] = useState('');
//   const [language, setLanguage] = useState('en'); // Default to English
//   const [loading, setLoading] = useState(false); // New state for loading

//   const handleSubmit = async (e) => {
//     e.preventDefault(); // Prevent default form submission behavior

//     setLoading(true); // Set loading state to true
//     setError(''); // Clear previous errors
//     setResponse(''); // Clear previous response

//     try {
//       const res = await axios.post('http://127.0.0.1:5000/process-query', {
//         user_question: userQuestion
//       });

//       const generatedResponse = res.data.generated_response;
//       if (generatedResponse === "I'm sorry, I cannot answer this question based on the provided context.") {
//         setError('');
//       }
//       setResponse(generatedResponse);

//       // Translate response
//       const translationRes = await axios.post('http://127.0.0.1:8000/translate', {
//         text: generatedResponse,
//         to: language
//       });
//       setTranslatedResponse(translationRes.data.translatedText);

//       // Save query and response to MongoDB
//       await axios.post('http://localhost:4000/save-query', {
//         userQuestion,
//         response: translationRes.data.translatedText
//       });
//     } catch (error) {
//       console.error('Error processing query:', error);
//       setError('Error: Unable to get response');
//     }

//     setLoading(false); // Set loading state to false
//   };

//   const handleLanguageChange = (e) => {
//     setLanguage(e.target.value);
//   };
//   const handleTextToSpeech = () => {
//     const utterance = new SpeechSynthesisUtterance(translatedResponse);
//     window.speechSynthesis.speak(utterance);
//   };



//   return (
//     <div className="query-form-container">
//       <form className="query-form" onSubmit={handleSubmit}>
//         <input
//           type="text"
//           value={userQuestion}
//           onChange={(e) => setUserQuestion(e.target.value)}
//           placeholder="Enter your question..."
//         />
//         <div className="language-selection">
//           <select value={language} onChange={handleLanguageChange}>
//             <option value="en">English</option>
//             <option value="hi">Hindi</option>
//             <option value="te">Telugu</option>
//             <option value="fr">French</option>
//             <option value="es">Spanish</option>
//             <option value="de">German</option>
//             <option value="it">Italian</option>
//             <option value="ja">Japanese</option>
//             <option value="ko">Korean</option>
//             <option value="zh-CN">Chinese (Simplified)</option>
//             <option value="zh-TW">Chinese (Traditional)</option>
//             <option value="ru">Russian</option>
//             <option value="ar">Arabic</option>
//             <option value="pt">Portuguese</option>
//             <option value="nl">Dutch</option>
//             <option value="sv">Swedish</option>
//             <option value="pl">Polish</option>
//             <option value="tr">Turkish</option>
//             <option value="vi">Vietnamese</option>
//             <option value="th">Thai</option>
//             <option value="el">Greek</option>
//           </select>
//           <button type="submit">Ask Question</button>
//         </div>
//       </form>
//       {error && <div className="error-message">{error}</div>}
//       {loading ? (
//         <div className="processing-message">Processing...</div>
//       ) : (
//         response && (
//           <div>
//             <button onClick={handleTextToSpeech}>Speak</button>
//             <div className="query-response">
//               {translatedResponse}
//             </div>
//           </div>
//         )
//       )}
//     </div>
//   );
// };

// export default QueryForm;

import React, { useState } from 'react';
import axios from 'axios';
import './QueryForm.css';

const QueryForm = () => {
  const [userQuestion, setUserQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [translatedResponse, setTranslatedResponse] = useState('');
  const [error, setError] = useState('');
  const [language, setLanguage] = useState('en'); 
  const [loading, setLoading] = useState(false);
  const [newLanguage, setNewLanguage] = useState('en'); 

  const handleSubmit = async (e) => {
    e.preventDefault(); 

    setLoading(true); 
    setError(''); 
    setResponse('');

    try {
      const res = await axios.post('http://127.0.0.1:5000/process-query', {
        user_question: userQuestion
      });

      const generatedResponse = res.data.generated_response;
      if (generatedResponse === "I'm sorry, I cannot answer this question based on the provided context.") {
        setError('');
      }
      setResponse(generatedResponse);

      
      const translationRes = await axios.post('http://127.0.0.1:8000/translate', {
        text: generatedResponse,
        to: language
      });
      setTranslatedResponse(translationRes.data.translatedText);

      // Save query and response to MongoDB
      await axios.post('http://localhost:4000/save-query', {
        userQuestion,
        response: translationRes.data.translatedText
      });
    } catch (error) {
      console.error('Error processing query:', error);
      setError('Error: Unable to get response');
    }

    setLoading(false); // Set loading state to false
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleNewLanguageChange = async (e) => {
    const newLang = e.target.value;
    setNewLanguage(newLang);
    try {
      const translationRes = await axios.post('http://127.0.0.1:8000/translate', {
        text: response,
        to: newLang
      });
      setTranslatedResponse(translationRes.data.translatedText);
    } catch (error) {
      console.error('Error translating response:', error);
      setError('Error: Unable to translate response');
    }
  };

  const handleTextToSpeech = () => {
    const utterance = new SpeechSynthesisUtterance(translatedResponse);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="query-form-container">
      <form className="query-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={userQuestion}
          onChange={(e) => setUserQuestion(e.target.value)}
          placeholder="Enter your question..."
        />
        <div className="language-selection">
          <select value={language} onChange={handleLanguageChange}>
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="te">Telugu</option>
            <option value="fr">French</option>
            <option value="es">Spanish</option>
            <option value="de">German</option>
            <option value="it">Italian</option>
            <option value="ja">Japanese</option>
            <option value="ko">Korean</option>
            <option value="zh-CN">Chinese (Simplified)</option>
            <option value="zh-TW">Chinese (Traditional)</option>
            <option value="ru">Russian</option>
            <option value="ar">Arabic</option>
            <option value="pt">Portuguese</option>
            <option value="nl">Dutch</option>
            <option value="sv">Swedish</option>
            <option value="pl">Polish</option>
            <option value="tr">Turkish</option>
            <option value="vi">Vietnamese</option>
            <option value="th">Thai</option>
            <option value="el">Greek</option>
          </select>
          <button type="submit">Ask Question</button>
        </div>
      </form>
      {error && <div className="error-message">{error}</div>}
     
      {loading ? (
       
        <div className="processing-message">Processing...</div>
      ) : (
        response && (
          <div>
            <button onClick={handleTextToSpeech}>Speak</button>
            <div className="new-language-selection">
              <select value={newLanguage} onChange={handleNewLanguageChange}>
                <option value="en">English</option>
                <option value="hi">Hindi</option>
                <option value="te">Telugu</option>
                <option value="fr">French</option>
                <option value="es">Spanish</option>
                <option value="de">German</option>
                <option value="it">Italian</option>
                <option value="ja">Japanese</option>
                <option value="ko">Korean</option>
                <option value="zh-CN">Chinese (Simplified)</option>
                <option value="zh-TW">Chinese (Traditional)</option>
                <option value="ru">Russian</option>
                <option value="ar">Arabic</option>
                <option value="pt">Portuguese</option>
                <option value="nl">Dutch</option>
                <option value="sv">Swedish</option>
                <option value="pl">Polish</option>
                <option value="tr">Turkish</option>
                <option value="vi">Vietnamese</option>
                <option value="th">Thai</option>
                <option value="el">Greek</option>
              </select>
            </div>
            <div className='res'>
              <div className="query-response">
              {translatedResponse}
              </div>
            </div>
            
          </div>

        )

      )}
    </div>
  );
};

export default QueryForm;
