import React, { useRef, useState, useEffect } from 'react';
import Webcam from "react-webcam";
import { 
  Container, 
  Button, 
  Typography, 
  Paper, 
  Box, 
  Slider,
  CircularProgress,
  Chip,
  Stack,
  Alert
} from '@mui/material';
import { 
  SentimentVeryDissatisfied, 
  SentimentDissatisfied, 
  SentimentNeutral, 
  SentimentSatisfied, 
  SentimentVerySatisfied 
} from '@mui/icons-material';

const emotionIcons = {
  'Fear': <SentimentVeryDissatisfied fontSize="large" color="error" />,
  'Anger': <SentimentVeryDissatisfied fontSize="large" color="error" />,
  'Sadness': <SentimentDissatisfied fontSize="large" color="warning" />,
  'Neutral': <SentimentNeutral fontSize="large" color="info" />,
  'Happiness': <SentimentSatisfied fontSize="large" color="success" />,
  'Surprise': <SentimentVerySatisfied fontSize="large" color="success" />,
};

function App() {
  // Refs
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameId = useRef(null);
  const active = useRef(false);
  const lastAnalysisTime = useRef(0);

  // State
  const [capturing, setCapturing] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [resultsHistory, setResultsHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisInterval] = useState(2000); // 2 seconds between analyses

  // Capture and analyze frames
  useEffect(() => {
    active.current = capturing;
    
    const captureFrame = async (timestamp) => {
      if (!active.current || !webcamRef.current) {
        return;
      }

      // Draw frame to canvas
      const canvas = canvasRef.current;
      const context = canvas.getContext('2d');
      context.drawImage(webcamRef.current.video, 0, 0, canvas.width, canvas.height);

      // Throttle analysis based on time
      const now = Date.now();
      if (now - lastAnalysisTime.current > analysisInterval) {
        lastAnalysisTime.current = now;
        
        try {
          setLoading(true);
          setError(null);
          
          const blob = await new Promise(resolve => 
            canvas.toBlob(resolve, 'image/jpeg', 0.9)
          );
          
          const formData = new FormData();
          formData.append('image', blob, 'current_frame.jpg');
          
          const response = await fetch('http://localhost:5000/save-image', {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
          }
          
          const data = await response.json();
          
          if (data.status === 'error') {
            throw new Error(data.message);
          }
          
          if (data.emotion && data.valence !== undefined && data.arousal !== undefined) {
            const newResult = {
              emotion: data.emotion,
              valence: parseFloat(data.valence),
              arousal: parseFloat(data.arousal),
              timestamp: new Date().toLocaleTimeString()
            };
            
            setCurrentResult(newResult);
            setResultsHistory(prev => [newResult, ...prev.slice(0, 4)]);
          }
        } catch (error) {
          console.error('Analysis error:', error);
          setError(error.message);
        } finally {
          setLoading(false);
        }
      }

      if (active.current) {
        animationFrameId.current = requestAnimationFrame(captureFrame);
      }
    };

    if (capturing) {
      lastAnalysisTime.current = 0; // Reset timer on start
      animationFrameId.current = requestAnimationFrame(captureFrame);
    }

    return () => {
      active.current = false;
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [capturing, analysisInterval]);

  const toggleCapture = () => {
    setCapturing(prev => !prev);
    setError(null);
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={6} sx={{ p: 3, borderRadius: 2 }}>
        <Stack spacing={3}>
          {/* Webcam Section */}
          <Typography variant="h4" align="center" color="primary">
            Emotion Analysis
          </Typography>
          
          <Webcam
            audio={false}
            ref={webcamRef}
            width="100%"
            height="auto"
            style={{ 
              borderRadius: 8, 
              border: '2px solid',
              borderColor: capturing ? '#1976d2' : '#e0e0e0'
            }}
            screenshotFormat="image/jpeg"
          />

          <Button
            variant="contained"
            size="large"
            color={capturing ? "error" : "primary"}
            onClick={toggleCapture}
            sx={{ py: 1.5 }}
            startIcon={loading ? <CircularProgress size={24} color="inherit" /> : null}
          >
            {capturing ? "Stop Analysis" : "Start Analysis"}
          </Button>

          {/* Error Display */}
          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {/* Current Results */}
          <Paper elevation={2} sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Current Analysis
            </Typography>
            
            {currentResult ? (
              <Stack spacing={3}>
                <Box textAlign="center">
                  <Chip
                    label={currentResult.emotion}
                    icon={emotionIcons[currentResult.emotion] || <SentimentNeutral />}
                    sx={{ 
                      fontSize: '1.2rem',
                      padding: 2,
                      bgcolor: 
                        currentResult.emotion === 'Happiness' ? 'success.light' :
                        currentResult.emotion === 'Fear' ? 'error.light' :
                        currentResult.emotion === 'Sadness' ? 'warning.light' : 'info.light'
                    }}
                  />
                  <Typography variant="caption" display="block" mt={1}>
                    Last updated: {currentResult.timestamp}
                  </Typography>
                </Box>

                <Box>
                  <Typography gutterBottom>
                    Valence: {currentResult.valence.toFixed(3)}
                    <Typography component="span" variant="caption" color="text.secondary" ml={1}>
                      (Negative ↔ Positive)
                    </Typography>
                  </Typography>
                  <Slider
                    value={currentResult.valence}
                    min={-1}
                    max={1}
                    step={0.01}
                    sx={{ width: '100%' }}
                    color={currentResult.valence > 0 ? "success" : "error"}
                  />
                </Box>

                <Box>
                  <Typography gutterBottom>
                    Arousal: {currentResult.arousal.toFixed(3)}
                    <Typography component="span" variant="caption" color="text.secondary" ml={1}>
                      (Calm ↔ Excited)
                    </Typography>
                  </Typography>
                  <Slider
                    value={currentResult.arousal}
                    min={0}
                    max={1}
                    step={0.01}
                    sx={{ width: '100%' }}
                    color={currentResult.arousal > 0.5 ? "warning" : "success"}
                  />
                </Box>
              </Stack>
            ) : (
              <Typography color="text.secondary" textAlign="center" py={2}>
                {capturing ? "Analyzing first frame..." : "No data yet. Start analysis to begin."}
              </Typography>
            )}
          </Paper>

          {/* Previous Results */}
          {resultsHistory.length > 0 && (
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Analysis History (Last 4)
              </Typography>
              <Stack spacing={1}>
                {resultsHistory.map((result, index) => (
                  <Box 
                    key={index} 
                    sx={{ 
                      p: 1.5, 
                      bgcolor: 'background.default', 
                      borderRadius: 1,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      borderLeft: '4px solid',
                      borderColor: result.valence > 0 ? 'success.main' : 'error.main'
                    }}
                  >
                    <Box>
                      <Typography variant="body2" fontWeight="medium">
                        {result.emotion}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {result.timestamp}
                      </Typography>
                    </Box>
                    <Typography variant="caption">
                      V: {result.valence.toFixed(2)}, A: {result.arousal.toFixed(2)}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Paper>
          )}
        </Stack>

        {/* Hidden canvas */}
        <canvas 
          ref={canvasRef} 
          style={{ display: 'none' }} 
          width={640} 
          height={480} 
        />
      </Paper>
    </Container>
  );
}

export default App;