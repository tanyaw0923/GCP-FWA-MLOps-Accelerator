SELECT prediction,count(*) 
FROM `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T12_58_14_815Z_669` 
group by prediction
#178/22


SELECT prediction,count(*) 
FROM `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T13_08_45_116Z_589` 
group by prediction
#176/24

#check all the tables created in the schema
SELECT
  table_name,
  creation_time
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.INFORMATION_SCHEMA.TABLES`
ORDER BY
  creation_time DESC;


SELECT
  COUNT(*) AS prediction_rows
FROM
  `fwa-mlops-accelerator-demo.fraud_experiments.predictions_2026_09_28T13_08_45_116Z_589`;
