const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const predictionRoutes = require('./routes/predictionRoutes');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json());

// Routes
app.use('/api', predictionRoutes);

app.get('/', (req, res) => {
  res.send('ChurnIQ Backend is running');
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
