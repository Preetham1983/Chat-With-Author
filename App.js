import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import QueryForm from './components/QueryForm';
import Response from './components/Response';
import QueryList from './components/QueryList'; 
import './App.css';

function App() {
  const [userQuestion, setUserQuestion] = useState('');
  const [responseLanguage, setResponseLanguage] = useState('en');

  return (
    <div>
      <div className="navbar">
        <h1>Chat With Author <span role="img" aria-label="chat">🤖</span></h1>
      </div>
      <div className='content-container'>
        <FileUpload />
        {userQuestion && <Response userQuestion={userQuestion} responseLanguage={responseLanguage}/>}
        <QueryForm setUserQuestion={setUserQuestion} setResponseLanguage={setResponseLanguage} />
        <QueryList /> 
      </div>
    </div>
  );
}

export default App;
