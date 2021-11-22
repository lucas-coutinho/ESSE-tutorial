export CURRENT_USER=$(id -u):$(id -g) 
export MEM_LIMIT=$((8*1024*1024*1024)) 
export CPU_LIMIT=$(nproc) 
docker-compose up