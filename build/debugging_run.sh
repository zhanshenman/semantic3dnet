../bin/point_cloud_main ../data/benchmark/bildstein_station1_xyz_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/bildstein_station3_xyz_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/bildstein_station5_xyz_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/domfountain_station1_xyz_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/domfountain_station2_xyz_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/domfountain_station3_xyz_intensity_rgb_train.txt
#../bin/point_cloud_main ../data/benchmark/neugasse_station1_xyz_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/sg27_station1_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/sg27_station2_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/sg27_station4_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/sg27_station5_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/sg27_station9_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/sg28_station4_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/untermaederbrunnen_station1_xyz_intensity_rgb_train.txt
../bin/point_cloud_main ../data/benchmark/untermaederbrunnen_station3_xyz_intensity_rgb_train.txt

cd ../data/benchmark/
mv sg27_station2_intensity_rgb_train.txt_aggregated.txt sg27_station2_intensity_rgb_valid.txt_aggregated.txt
rm train_all.txt_aggregated.txt
cat *_train.txt_aggregated.txt >> train_all.txt_aggregated.txt
