from app import app
from flaskext.mysql import MySQL

mysqls = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'admin'
app.config['MYSQL_DATABASE_PASSWORD'] = 'XiaoQingWa23319!'
app.config['MYSQL_DATABASE_DB'] = 'PREDICTION_DATABASE'
app.config['MYSQL_DATABASE_HOST'] = 'aws-simplified.cds5bp6fq21g.us-east-1.rds.amazonaws.com'
mysqls.init_app(app)