from pyexpat import ExpatError
from pytubefix import YouTube
import sys
import os

def on_complete(stream, file_path):
    print(stream)
    print(file_path)

def on_progress(stream, chunk, bytes_remaining):
    print(100 - (bytes_remaining / stream.filesize * 100))
    
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Already folder exists')

def main():
    cnt=0
    
    for i in link:
        if i[0]=='#':
            cnt+=1
            continue
        
        try:
            # Youtube객채 생성
            yt = YouTube(url=i, on_complete_callback=on_complete, on_progress_callback=on_progress)
            yt_stream = yt.streams
        except:
            print('Connection Error')
        # 모든 파일 mp4 저장으로 설정
        mp4files = yt_stream.filter(file_extension=format, res = res, video_codec=codec)
        if len(mp4files)==0:
            raise Exception('Not found video')
        
        my_stream = mp4files.order_by('resolution').desc().first()
        
        try:
            makedirs(os.path.join(SAVE_PATH, str(cnt)))
            # 비디오 다운받기
            my_stream.download(os.path.join(SAVE_PATH, str(cnt)))
            cnt+=1
        except: 
            print("에러 발생!")
    print('다운 완료!')

if __name__ == '__main__':
    SAVE_PATH = '/home/byeooon/HDD/gurka_test' # 경로 설정 
    link= open('/home/byeooon/Desktop/gurka_data/youtube.txt', 'r') # 다운로드할 유튜브 링크 리스트에 담기
    
    format = 'mp4'
    res = '720p'
    codec = 'avc1.4d4020' # 'avc1.640020' # 'avc1.4d4020'
    
    main()