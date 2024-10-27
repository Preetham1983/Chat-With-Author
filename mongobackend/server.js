const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = 4000;

app.use(bodyParser.json());
app.use(cors());


mongoose.connect('mongodb://127.0.0.1:27017/queries', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const db = mongoose.connection;
db.on('error', console.error.bind(console, 'connection error:'));
db.once('open', () => {
  console.log('Connected to MongoDB');
});

// Define a schema and model for the queries
const querySchema = new mongoose.Schema({
  userQuestion: String,
  response: String,
});

const Query = mongoose.model('Query', querySchema);

app.post('/save-query', async (req, res) => {
  const { userQuestion, response } = req.body;
  
  const newQuery = new Query({ userQuestion, response });
  
  try {
    await newQuery.save();
    res.status(201).send(newQuery);
  } catch (error) {
    res.status(400).send('Error saving query');
  }
});


// Endpoint to get all queries
app.get('/getqueries', async (req, res) => {
  try {
    const queries = await Query.find();
    res.status(200).send(queries);
  } catch (error) {
    res.status(500).send('Error fetching queries');
  }
});

app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
