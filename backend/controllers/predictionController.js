const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

exports.predictChurn = (req, res) => {
  const inputData = req.body;

  if (!inputData) {
    return res.status(400).json({ error: 'No input data provided' });
  }

  // Path to predict.py script
  const scriptPath = path.join(__dirname, '../../ml-service/predict.py');
  const projectRoot = path.join(__dirname, '../..');
  const mlServiceDir = path.join(projectRoot, 'ml-service');
  const venvPython = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');

  // Prefer an explicit env var, then local venv, then system python.
  const pythonExecutable = process.env.PYTHON_EXECUTABLE
    || (fs.existsSync(venvPython) ? venvPython : 'python');

  // Call python script
  const pythonProcess = spawn(
    pythonExecutable,
    [scriptPath, JSON.stringify(inputData)],
    { cwd: mlServiceDir }
  );

  let resultData = '';
  let errorData = '';
  let responded = false;

  pythonProcess.on('error', (err) => {
    if (responded) return;
    responded = true;
    console.error('Failed to start Python process:', err.message);
    return res.status(500).json({
      error: 'Prediction failed',
      details: `Python process failed to start: ${err.message}`,
    });
  });

  pythonProcess.stdout.on('data', (data) => {
    resultData += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    errorData += data.toString();
  });

  pythonProcess.on('close', (code) => {
    if (responded) return;
    if (code !== 0) {
      responded = true;
      console.error(`Python script error (code ${code}):`, errorData);
      return res.status(500).json({ error: 'Prediction failed', details: errorData });
    }

    try {
      const prediction = JSON.parse(resultData);
      responded = true;
      res.status(200).json(prediction);
    } catch (parseError) {
      responded = true;
      console.error('Failed to parse prediction result:', resultData);
      res.status(500).json({
        error: 'Invalid prediction output format',
        details: resultData || String(parseError),
      });
    }
  });
};
