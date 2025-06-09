function output = f_load_data_from_mat(file_path)

% If you have any question, please email to jangwon@usc.edu
% Jangwon Kim
% May 2nd 2014

% 공백이 있는 경로를 처리하기 위해 경로를 따옴표로 감싸기
file_path1 = strrep(file_path,'._','');
escaped_Path = strrep(file_path1, ' ', ' ');
command = ['chmod u+rw "', escaped_Path, '"']; % 경로를 " "로 감싸기
status = system(command);

if status == 0
    disp('파일의 읽기/쓰기 권한이 성공적으로 부여되었습니다.');
else
    disp('권한 부여에 실패했습니다.');
end
tmp_data = load(escaped_Path);
tmp_fieldname = fieldnames(tmp_data);
output = getfield(tmp_data,tmp_fieldname{1});

