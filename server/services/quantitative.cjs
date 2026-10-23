/**
 * Quantitative Analysis Agent
 * Performs deterministic financial calculations
 */

// Simple hash function to generate consistent values for the same query
function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

// Generate deterministic but varied financial metrics based on query
function generateMetrics(query) {
  const hash = simpleHash(query.toLowerCase().trim());

  // Use hash to generate consistent but varied values
  // Revenue growth: -15% to +35%
  const revenueGrowth = ((hash % 50) - 15) / 100;

  // Profit margin: 2% to 25%
  const profitMargin = ((hash % 23) + 2) / 100;

  // Liquidity ratio: 0.8 to 3.5
  const liquidityRatio = 0.8 + ((hash % 27) / 10);

  // Leverage ratio: 0.3 to 2.5
  const leverageRatio = 0.3 + ((hash % 22) / 10);

  return {
    revenueGrowth,
    profitMargin,
    liquidityRatio,
    leverageRatio,
  };
}

function analyze(query) {
  // For the MVP, generate company-specific synthetic metrics
  // In a production system, this would parse actual financial statements

  const metrics = generateMetrics(query || 'default');

  return {
    revenueGrowth: metrics.revenueGrowth,
    profitMargin: metrics.profitMargin,
    liquidityRatio: metrics.liquidityRatio,
    leverageRatio: metrics.leverageRatio,
    reportingPeriod: 'FY 2024 (Estimated)',
    currency: 'USD (Estimated)',
    isSynthetic: true,
    note: 'These are synthetic estimates based on the company name. In production, this would parse actual financial statements from official filings.',
  };
}

module.exports = { analyze };
