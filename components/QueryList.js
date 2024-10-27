import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './QueryList.css';

const QueryList = () => {
  const [queries, setQueries] = useState([]);
  const [selectedQuery, setSelectedQuery] = useState(null);

  useEffect(() => {
    const fetchQueries = async () => {
      try {
        const res = await axios.get('http://localhost:4000/getqueries');
        setQueries(res.data);
      } catch (error) {
        console.error('Error fetching queries:', error);
      }
    };

    fetchQueries();
  }, []);

  const handleQueryClick = (query) => {
    setSelectedQuery(query);
  };

  return (
    <div className="chat-container">
      <div className="query-list-container">
      <h2 style={{ color: 'rgb(173, 216, 230)' }}>Stored Queries</h2>

        <div className="query-list">
          {queries.map((query) => (
            <div key={query._id} className="query-item" onClick={() => handleQueryClick(query)}>
              <div className="question">
                <div className="question-text">{query.userQuestion}</div>
               
              </div>
            </div>
          ))}
        </div>
      </div>
      {selectedQuery && (
        <div className="response-container">
          <div className="response-display">
            <strong>Response:</strong>
            <div className="response-text">{selectedQuery.response}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QueryList;

