-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Schemas
CREATE SCHEMA IF NOT EXISTS mia;       -- application tables
CREATE SCHEMA IF NOT EXISTS xbrl;      -- structured EDGAR/XBRL financial facts
CREATE SCHEMA IF NOT EXISTS eval;      -- evaluation golden set + results
