clear all; clc;

% 폴더 내 모든 파일에 대해 작업을 자동으로 수행하는 MATLAB 코드
run("config/options.m");
run("config/paths.m");
addpath('code_fn/')
addpath('code_wrapper/')
addpath('script/');


if ~exist(path.output_data_dir,'dir')
    system(['mkdir -p "' path.output_data_dir '"']); % 결과 데이터 디렉토리가 없으면 생성
end

if ~exist(path.morph_data_dir,'dir')
    system(['mkdir -p "' path.morph_data_dir '"']); % 결과 데이터 디렉토리가 없으면 생성
end

if ~exist(path.sil_lab,'dir')
    system(['mkdir -p "' path.sil_lab '"']); % 결과 데이터 디렉토리가 없으면 생성
end

if ~exist(path.mri_data_dir,'dir')
    system(['mkdir -p "' path.mri_data_dir '"']);
end


% Step 1: 데이터 준비 (AVI → MAT 변환)
run('script/conv_avi_mat.m');


% Step 2: 그리드 생성 (MAT 파일 기반)
run("code_wrapper/wrapper_grid.m"); % 그리드 생성 함수 호출


% Step 3: 음성 관로 매개변수 추출
run('code_wrapper/wrapper_seg_batch'); % 매개변수 추출 함수 호출


% Step 4: 형태학적 매개변수 계산
run('code_wrapper/wrapper_morph'); % 형태학적 매개변수 계산 함수 호출


fprintf('All files processed successfully.');