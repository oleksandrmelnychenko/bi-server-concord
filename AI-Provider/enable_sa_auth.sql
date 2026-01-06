-- Run this script in SQL Server Management Studio (SSMS) as Administrator
-- Step 1: Enable Mixed Mode Authentication (requires SSMS GUI or registry change)
-- Step 2: Enable and configure SA account

-- Enable the SA login
ALTER LOGIN sa ENABLE;
GO

-- Set the SA password
ALTER LOGIN sa WITH PASSWORD = '1234';
GO

-- Verify SA is enabled
SELECT name, is_disabled, type_desc
FROM sys.server_principals
WHERE name = 'sa';
GO

-- After running this script:
-- 1. Open SQL Server Configuration Manager
-- 2. Right-click your SQL Server instance -> Properties -> Security
-- 3. Select "SQL Server and Windows Authentication mode"
-- 4. Restart SQL Server service
