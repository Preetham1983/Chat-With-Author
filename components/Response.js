import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Response.css';
const Response = ({ userQuestion, responseLanguage }) => {
  const [generatedResponse, setGeneratedResponse] = useState('');
  // const [loading, setLoading] = useState(false);
  // const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      // setLoading(true);
      // setError(null);

      try {
        const response = await axios.post('http://localhost:5000/process-query',{
          user_question: userQuestion,
          response_language: responseLanguage,
        });

        setGeneratedResponse(response.data.generated_response);
      } catch (error) {
        
        console.error('Error fetching response:', error);
      } finally {
        // setLoading(false);
      }
    };

    fetchData();
  }, [userQuestion, responseLanguage]);

  // if (loading) {
  //   return <p>Loading...</p>;
  // }

  // if (error) {
  //   return <p>{error}</p>;
  // }

  return (
    <div>
      <h2>Generated Response:</h2>
      <p>{generatedResponse}</p>
    </div>
  );
};

export default Response;
