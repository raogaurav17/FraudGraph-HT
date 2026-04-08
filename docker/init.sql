-- FraudGraph database initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Indexes for analytics queries
DO $$
BEGIN
	IF to_regclass('public.fraud_predictions') IS NOT NULL THEN
		CREATE INDEX IF NOT EXISTS idx_predictions_created ON fraud_predictions(created_at DESC);
	END IF;

	IF to_regclass('public.transactions') IS NOT NULL THEN
		CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);
	END IF;
END
$$;
