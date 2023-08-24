temperatures=(1 10 50 100 150 200 250 300 350 400 450 500)

for temp in "${temperatures[@]}"; do
    python position-prediction.py --load_directory "test$temp" --temperature "$temp" &&
    echo "Successfully trained T=$temp and calculated MSE,RMSE"
done

echo "ALL DONE!"
