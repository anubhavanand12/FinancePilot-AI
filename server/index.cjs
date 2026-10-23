const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const researcher = require('./agents/researcher.cjs');
const reviewer = require('./agents/reviewer.cjs');
const quantitative = require('./services/quantitative.cjs');
const mlClient = require('./services/ml-client.cjs');

const app = express();
const PORT = process.env.PORT || 3001;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Initialize Gemini
const geminiApiKey = process.env.GEMINI_API_KEY;
let genAI = null;
let model = null;

if (geminiApiKey) {
  try {
    genAI = new GoogleGenerativeAI(geminiApiKey);
    model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
    console.log('Gemini initialized successfully');
  } catch (error) {
    console.error('Failed to initialize Gemini:', error.message);
  }
} else {
  console.warn('GEMINI_API_KEY not set - AI features will be limited');
}

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    gemini: !!model,
    mlService: ML_SERVICE_URL,
    timestamp: new Date().toISOString(),
  });
});

// Demo endpoint (works without Gemini)
app.post('/api/analyze-demo', async (req, res) => {
  try {
    const agents = {
      researcher: { status: 'demo', message: 'Using preloaded demo data' },
      quantitative: { status: 'demo', message: 'Using preloaded demo data' },
      ml: { status: 'demo', message: 'Using preloaded demo data' },
      reviewer: { status: 'demo', message: 'Using preloaded demo data' },
    };

    // Demo financial snapshot - use quantitative agent for consistent generation
    const financialSnapshot = quantitative.analyze('Demo Company Example');

    // Demo ML prediction (placeholder - will be replaced with real ML call)
    let mlPrediction = null;
    try {
      const mlResponse = await fetch(`${ML_SERVICE_URL}/predict-demo`);
      if (mlResponse.ok) {
        mlPrediction = await mlResponse.json();
        agents.ml = { status: 'completed', message: 'ML prediction successful' };
      } else {
        throw new Error('ML service returned error');
      }
    } catch (error) {
      console.error('ML service error in demo mode:', error.message);
      agents.ml = { status: 'failed', message: error.message };
      // Fallback demo data
      mlPrediction = {
        model_type: 'RandomForestClassifier',
        prediction: 0,
        model_estimated_probability_of_default: 0.23,
        probability_is_calibrated: false,
        model_version: 'v1',
        dataset: 'UCI Default of Credit Card Clients',
        limitations: [
          'Historical Taiwanese dataset from 2005',
          'Not validated for Indian borrowers',
          'Demo data - not a real prediction',
        ],
        feature_importance: {
          'PAY_0': 0.25,
          'PAY_2': 0.18,
          'PAY_3': 0.15,
          'LIMIT_BAL': 0.12,
          'BILL_AMT1': 0.10,
        },
      };
    }

    // Demo research report - dynamically reflect the financial metrics
    const researchReport = {
      mainFindings: [
        'This is a demonstration of the FinancePilot AI system',
        'The system combines AI research, quantitative analysis, and ML predictions',
        'Demo mode uses preloaded data for testing without live APIs',
      ],
      positiveEvidence: [
        `Revenue growth of ${(financialSnapshot.revenueGrowth * 100).toFixed(1)}% year-over-year`,
        `Profit margin of ${(financialSnapshot.profitMargin * 100).toFixed(1)}%`,
        `Liquidity ratio of ${financialSnapshot.liquidityRatio.toFixed(2)}`,
      ],
      negativeEvidence: [
        `Leverage ratio of ${financialSnapshot.leverageRatio.toFixed(2)}`,
        'Market volatility presents risks',
      ],
      missingInformation: [
        'Current market conditions',
        'Competitor analysis',
        'Management quality assessment',
      ],
      riskFactors: [
        'Economic uncertainty',
        'Regulatory changes',
        'Market competition',
      ],
      reviewerComments: [
        'This is a demo report - not based on real company data',
        'For demonstration purposes only',
        'Do not use for actual financial decisions',
      ],
      citations: [
        { text: 'Demo mode - no external sources accessed', url: null },
      ],
      disclaimer: 'This is a demonstration using synthetic data. Not for actual financial decision-making.',
    };

    // Demo evaluation metrics
    const evaluation = {
      accuracy: 0.8195,
      precision_default: 0.6623,
      recall_default: 0.3692,
      f1_score: 0.4737,
      roc_auc: 0.7623,
      average_precision: 0.4156,
    };

    res.json({
      agents,
      financialSnapshot,
      mlPrediction,
      researchReport,
      evaluation,
    });
  } catch (error) {
    console.error('Demo analysis error:', error);
    res.status(500).json({
      error: 'Demo analysis failed',
      message: error.message,
    });
  }
});

// Main analysis endpoint
app.post('/api/analyze', async (req, res) => {
  const { query } = req.body;

  if (!query || typeof query !== 'string') {
    return res.status(400).json({
      error: 'Invalid request',
      message: 'Query is required and must be a string',
    });
  }

  const agents = {
    researcher: { status: 'pending' },
    quantitative: { status: 'pending' },
    ml: { status: 'pending' },
    reviewer: { status: 'pending' },
  };

  try {
    // Step 1: Quantitative Analysis (deterministic)
    agents.quantitative = { status: 'running' };
    let financialSnapshot = null;
    try {
      financialSnapshot = quantitative.analyze(query);
      agents.quantitative = {
        status: 'completed',
        message: 'Financial ratios calculated',
      };
    } catch (error) {
      agents.quantitative = {
        status: 'failed',
        message: error.message,
      };
    }

    // Step 2: ML Risk Assessment
    agents.ml = { status: 'running' };
    let mlPrediction = null;
    try {
      const mlResponse = await fetch(`${ML_SERVICE_URL}/predict-demo`);
      if (mlResponse.ok) {
        mlPrediction = await mlResponse.json();
        agents.ml = { status: 'completed', message: 'ML prediction successful' };
      } else {
        throw new Error('ML service returned error');
      }
    } catch (error) {
      console.error('ML service error:', error.message);
      agents.ml = { status: 'failed', message: error.message };
    }

    // Step 3: Financial Research (Gemini)
    agents.researcher = { status: 'running' };
    let researchReport = null;
    if (model) {
      try {
        researchReport = await researcher.analyze(query, model, financialSnapshot);
        agents.researcher = {
          status: 'completed',
          message: 'Research completed',
        };
      } catch (error) {
        console.error('Research agent error:', error.message);
        agents.researcher = {
          status: 'failed',
          message: error.message,
        };
      }
    } else {
      agents.researcher = {
        status: 'skipped',
        message: 'Gemini not configured',
      };
    }

    // Step 4: Independent Review (Gemini)
    agents.reviewer = { status: 'running' };
    if (model && researchReport) {
      try {
        const reviewResult = await reviewer.review(
          query,
          researchReport,
          financialSnapshot,
          model
        );
        researchReport.reviewerComments = reviewResult.comments;
        agents.reviewer = {
          status: 'completed',
          message: 'Review completed',
        };
      } catch (error) {
        console.error('Reviewer agent error:', error.message);
        agents.reviewer = {
          status: 'failed',
          message: error.message,
        };
      }
    } else {
      agents.reviewer = {
        status: 'skipped',
        message: model ? 'No research report to review' : 'Gemini not configured',
      };
    }

    // Get evaluation metrics from ML service
    let evaluation = null;
    try {
      const metricsResponse = await fetch(`${ML_SERVICE_URL}/metrics`);
      if (metricsResponse.ok) {
        evaluation = await metricsResponse.json();
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error.message);
    }

    res.json({
      agents,
      financialSnapshot,
      mlPrediction,
      researchReport,
      evaluation,
    });
  } catch (error) {
    console.error('Analysis error:', error);
    res.status(500).json({
      error: 'Analysis failed',
      message: error.message,
      agents,
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`FinancePilot API server running on port ${PORT}`);
  console.log(`ML service URL: ${ML_SERVICE_URL}`);
  console.log(`Gemini ${model ? 'configured' : 'not configured'}`);
});
