# Set the port that your application uses
PORT=12345  # Change this to match the port your application uses

# A loop that runs indefinitely
while true
do
    echo "Checking for processes using port $PORT..."
    # Find and kill any processes using the specified port
    if lsof -i :$PORT -t &> /dev/null; then
        echo "Killing processes using port $PORT..."
        lsof -i :$PORT -t | xargs kill -9
        # Give some time for the port to be released
        sleep 5
    fi
    
    echo "Running server..."
    # Run the Isaac Sim Python script with your collection file
    roslaunch grasp_generation grasp_single.launch
    
    echo "Script finished. Resting for 20 seconds..."
    sleep 5  # 20 seconds
    
    # Loop repeats
done