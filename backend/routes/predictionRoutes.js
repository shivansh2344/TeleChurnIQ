const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/predictionController');

router.post('/predict', predictionController.predictChurn);
router.get('/predict', (req, res) => {
  res.json({ 
    message: 'Churn prediction endpoint is active. Use POST to send feature data.',
    endpoint: '/api/predict',
    method: 'POST'
  });
});

module.exports = router;
