import pymysql
from app import app
from config import mysqls
from flask import jsonify
from flask import flash, request

@app.route('/empty')
def empty():
    try:
        conn = mysqls.connect()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT * FROM sample")
        empRows = cursor.fetchall()
        response = jsonify(empRows)
        response.status_code = 200
        return response
    except Exception as e:
        print(e)
    finally:
        conn = mysqls.connect()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.close()
        conn.close()


@app.route('/emp/<int:Location>/<int:Instance>/<int:OS>/<int:day>/<int:month>/<int:year>')
def emp(Location,Instance,OS,day,month,year):
    try:
        conn = mysqls.connect()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT PricePerUnit FROM sample WHERE Location = %s and Instance = %s and OS = %s and day=%s and month=%s and year=%s", (Location, Instance, OS, day, month, year))
        empRow = cursor.fetchmany()
        response = jsonify(empRow)
        response.status_code = 200
        return response
    except Exception as e:
        print(e)
    finally:
        conn = mysqls.connect()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.close()
        conn.close()


@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Record not found: ' + request.url,
    }
    response = jsonify(message)
    response.status_code = 404
    return response


if __name__ == "__main__":
    app.debug = True
    app.run()