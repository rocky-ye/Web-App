# docker build -t create_databas .
# sh run_docker.sh

#export DATABASE_NAME=msia423_db

# docker run -it \
# --env MYSQL_HOST \
# --env MYSQL_PORT \
# --env MYSQL_USER \
# --env MYSQL_PASSWORD \
# --env DATABASE_NAME penny_mysql penny_lane_db.py 

# -p ${MYSQL_PORT}:${MYSQL_PORT} \ 
# echo "sup"
# echo  ${MYSQL_HOST}

# docker build -t insurance .
# docker run insurance run.py acquire_data
# docker run --mount type=bind,source="$(pwd)"/data,target=/app/data insurance src/create_database.py --local

docker run -it \
--env MYSQL_HOST \
--env MYSQL_PORT \
--env MYSQL_USER \
--env MYSQL_PASSWORD \
--env DATABASE_NAME \
insurance run.py create_db