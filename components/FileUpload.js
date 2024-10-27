import React, { useState } from 'react';
import axios from 'axios';
import './FileUpload.css';

const FileUpload = () => {
  const [selectedFiles, setSelectedFiles] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(false);

  const handleFileChange = (e) => {
    setSelectedFiles(e.target.files);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append('pdf_files', selectedFiles[i]);
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(false);

    try {
      await axios.post('http://127.0.0.1:5000/upload-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadSuccess(true);
    } catch (error) {
      console.error('Error uploading PDFs:', error);
      setUploadError('Error uploading PDFs');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="file-upload-container">
      <label className="file-upload-label">
        <input type="file" onChange={handleFileChange} multiple className="file-upload-input" />
        Choose File(s)
      </label>
      <button onClick={handleUpload} className="file-upload-button">
        Upload & Process
      </button>
     
      {uploading && <p className="upload-status">Uploading...</p>}
      {uploadSuccess && <p className="upload-status success">DONE</p>}
      {uploadError && <p className="upload-status error">Error uploading PDFs</p>}
    </div>
  );
};

export default FileUpload;


