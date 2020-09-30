# Known Issues
 - TailIndex only allows string values as attributes while psycopg2 properly
   type-casts values coming from Posgres DBs.
 - TailIndex can't index filtered attribute (e.g. `SELECT age, COUNT(*) GROUP BY age WHERE age=10;`)
