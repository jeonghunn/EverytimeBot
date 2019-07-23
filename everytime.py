"""Simple Bot to reply to Telegram messages.
This is built on the API wrapper, see echobot2.py to see the same example built
on the telegram.ext bot framework.
This program is dedicated to the public domain under the CC0 license.
"""
import logging
import telegram
import requests, json
import traceback
from time import sleep
import time
import tensorflow as tf
import data
import model as ml
from configs import DEFINES
import random
import math
import os, sys
import datetime
import xml.etree.ElementTree as elemTree
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from StopHook import StopHook
DATA_OUT_PATH = './data_out/'
DATA_OUT_CANDIDATE_PATH = './data_out_candidate/'

# Serving 기능을 위하여 serving 함수를 구성한다.
def serving_input_receiver_fn():
    receiver_tensor = {
        'input': tf.placeholder(dtype=tf.int32, shape=[None, DEFINES.max_sequence_length]),
        'output': tf.placeholder(dtype=tf.int32, shape=[None, DEFINES.max_sequence_length])
    }
    features = {
        key: tensor for key, tensor in receiver_tensor.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

def train(step):
    bot = telegram.Bot('auth')
    data_out_path = os.path.join(os.getcwd(), DATA_OUT_PATH)
    data_out_candidate_path = os.path.join(os.getcwd(), DATA_OUT_CANDIDATE_PATH)
    os.makedirs(data_out_path, exist_ok=True)
    os.makedirs(data_out_candidate_path, exist_ok=True)
    # 데이터를 통한 사전 구성 한다.
    char2idx, idx2char, vocabulary_length = data.load_vocabulary_for_train()
    # 훈련 데이터와 테스트 데이터를 가져온다.
    train_input, train_label, eval_input, eval_label = data.load_data_for_train()

    # 훈련셋 인코딩 만드는 부분이다.
    train_input_enc, train_input_enc_length = data.enc_processing(train_input, char2idx)
    # 훈련셋 디코딩 출력 부분 만드는 부분이다.
    train_target_dec, train_target_dec_length = data.dec_target_processing(train_label, char2idx)

    # 평가셋 인코딩 만드는 부분이다.
    eval_input_enc, eval_input_enc_length = data.enc_processing(eval_input,char2idx)
    # 평가셋 디코딩 출력 부분 만드는 부분이다.
    eval_target_dec, _ = data.dec_target_processing(eval_label, char2idx)

    # 현재 경로'./'에 현재 경로 하부에
    # 체크 포인트를 저장한 디렉토리를 설정한다.
    check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
    save_model_path = os.path.join(os.getcwd(), DEFINES.save_model_path)
    check_point_candidate_path = os.path.join(os.getcwd(), DEFINES.check_point_candidate_path)
    save_model_candidate_path = os.path.join(os.getcwd(), DEFINES.save_model_candidate_path)
    # 디렉토리를 만드는 함수이며 두번째 인자 exist_ok가
    # True이면 디렉토리가 이미 존재해도 OSError가
    # 발생하지 않는다.
    # exist_ok가 False이면 이미 존재하면
    # OSError가 발생한다.
    os.makedirs(check_point_candidate_path, exist_ok=True)
    os.makedirs(save_model_candidate_path, exist_ok=True)
    os.makedirs(check_point_path, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)




    # 에스티메이터 구성한다.
    classifier = tf.estimator.Estimator(
        model_fn=ml.Model,  # 모델 등록한다.
        model_dir=DEFINES.check_point_candidate_path,  # 체크포인트 위치 등록한다.
        params={  # 모델 쪽으로 파라메터 전달한다.
            'hidden_size': DEFINES.hidden_size,  # 가중치 크기 설정한다.
            'layer_size': DEFINES.layer_size,  # 멀티 레이어 층 개수를 설정한다.
            'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
            'teacher_forcing_rate': DEFINES.teacher_forcing_rate, # 학습시 디코더 인풋 정답 지원율 설정
            'vocabulary_length': vocabulary_length,  # 딕셔너리 크기를 설정한다.
            'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
            'embedding': DEFINES.embedding,  # 임베딩 사용 유무를 설정한다.
            'multilayer': DEFINES.multilayer,  # 멀티 레이어 사용 유무를 설정한다.
            'attention': DEFINES.attention, #  어텐션 지원 유무를 설정한다.
            'teacher_forcing': DEFINES.teacher_forcing, # 학습시 디코더 인풋 정답 지원 유무 설정한다.
            'loss_mask': DEFINES.loss_mask, # PAD에 대한 마스크를 통한 loss를 제한 한다.
            'serving': DEFINES.serving # 모델 저장 및 serving 유무를 설정한다.
        })


    earlystop = StopHook()

    # 학습 실행
    tf.estimator.train_and_evaluate(classifier,train_spec=tf.estimator.TrainSpec(input_fn=lambda: data.train_input_fn(train_input_enc, train_target_dec_length, train_target_dec, DEFINES.batch_size), max_steps=step, hooks=[earlystop]), eval_spec=tf.estimator.EvalSpec(input_fn=lambda: data.eval_input_fn(eval_input_enc,eval_target_dec, DEFINES.batch_size)))
    # 서빙 기능 유무에 따라 모델을 Save 한다.
    save_model_path = classifier.export_savedmodel(
    export_dir_base=DEFINES.save_model_path,
    serving_input_receiver_fn=serving_input_receiver_fn)

    # 평가 실행
    # eval_result = classifier.evaluate(input_fn=lambda: data.eval_input_fn(
    #     eval_input_enc,eval_target_dec, DEFINES.batch_size))
    # print('\nEVAL set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def learn():

    try:
        bot = telegram.Bot('auth')
        bot.send_message(chat_id = -116418298, text="[학습 시작]")

        # for the_file in os.listdir(folder):
        #     file_path = os.path.join(folder, the_file)
        #     try:
        #         if os.path.isfile(file_path):
        #             os.unlink(file_path)
        #         #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        #     except Exception as e:
        #         print(e)
        #         bot.send_message(chat_id = -116418298, text="[오류] 기존 모델을 삭제하는데 실패했습니다.")

        # with tf.Session() as sess:
        #     print("새로운 모델을 생성하는 중 입니다.")
        #
        #     sess.run(tf.global_variables_initializer())
        # for i in range(0, index):
        #     ii = i+1
        #     tf.reset_default_graph()
        #     bot.send_message(chat_id = -116418298, text=str(ii) +"번째 책에 대한 공부 진행 중...")
        #     print(str(ii) +"번째 책에 대한 공부 진행 중...")
        #     dialog = Dialog()
        #     dialog.load_vocab(FLAGS.voc_candidate_path)
        #     dialog.load_examples("./chat" + str(ii) + "000.log")
        #     trainbot(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
        # bot.send_message(chat_id = -116418298, text="오늘 할 공부가 다 끝났습니다. 적용을 위해 봇을 재시작합니다.")
        os.system("rm -rf data_out_candidate/*")
        train(10000)
        os.system("rm -rf  data_out; rm -rf data_in")
        os.system("mv data_out_candidate data_out")
        os.system("cp -R data_in_candidate data_in")

        bot.send_message(chat_id = -116418298, text="[학습 종료]")
        bootcount = 0
        os.system("nohup ./restart.sh")
        sleep(0.2) # 200ms to CTR+C twice
    except:
        bot.send_message(chat_id = -116418298, text="[오류] 학습을 진행하던 도중 문제가 발생하여 종료하였습니다. 점검이 필요합니다.\n" + traceback.format_exc())
        print(traceback.format_exc())


def learnquick():
    try:
        bot = telegram.Bot('auth')
        bot.send_message(chat_id = -116418298, text="[학습 프로세스 시작] 새로운 변경 내역을 학습합니다. 이 작업은 하루에 한번정도 실행됩니다.")

        # for the_file in os.listdir(folder):
        #     file_path = os.path.join(folder, the_file)
        #     try:
        #         if os.path.isfile(file_path):
        #             os.unlink(file_path)
        #         #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        #     except Exception as e:
        #         print(e)
        #         bot.send_message(chat_id = -116418298, text="[오류] 기존 모델을 삭제하는데 실패했습니다.")

        # with tf.Session() as sess:
        #     print("새로운 모델을 생성하는 중 입니다.")
        #
        #     sess.run(tf.global_variables_initializer())
        # for i in range(0, index):
        #     ii = i+1
        #     tf.reset_default_graph()
        #     bot.send_message(chat_id = -116418298, text=str(ii) +"번째 책에 대한 공부 진행 중...")
        #     print(str(ii) +"번째 책에 대한 공부 진행 중...")
        #     dialog = Dialog()
        #     dialog.load_vocab(FLAGS.voc_candidate_path)
        #     dialog.load_examples("./chat" + str(ii) + "000.log")
        #     trainbot(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
        # bot.send_message(chat_id = -116418298, text="오늘 할 공부가 다 끝났습니다. 적용을 위해 봇을 재시작합니다.")


        os.system("cp -R data_out data_out_candidate")
        train(2500)
        os.system("mv ./data_out_candidate data_out")
        os.system("rm -rf data_in")
        os.system("cp -R data_in_candidate data_in")


        bot.send_message(chat_id = -116418298, text="오늘 할 공부가 다 끝났습니다. 적용을 위해 봇을 재시작합니다.")
        bootcount = 0
        os.system("nohup ./restart.sh")
        sleep(0.2) # 200ms to CTR+C twice
    except:
        bot.send_message(chat_id = -116418298, text="[오류] 학습을 진행하던 도중 문제가 발생하여 종료하였습니다. 점검이 필요합니다.\n" + traceback.format_exc())
        print(traceback.format_exc())



def main():
    """Run the bot."""
global update_id

bot = telegram.Bot('auth')
clastid =0
lastid = 0
with open("./everytime.txt", "r") as f:
    lastid = int(f.readline()) # read everything in the file

while 1:
 try:




    loadcount = 0
    everytime_list_url = "http://everytime.kr/find/board/article/list"
    everytime_art_url = "https://everytime.kr/find/board/comment/list"
    everytime_commentw_url = "https://everytime.kr/save/board/comment"
    cookie = "에브리타임 쿠키 정보."
    board_id = "374911"


    # 데이터를 통한 사전 구성 한다.
    char2idx,  idx2char, vocabulary_length = data.load_vocabulary()





    # 에스티메이터 구성한다.
    classifier = tf.estimator.Estimator(
        model_fn=ml.Model, # 모델 등록한다.
        model_dir=DEFINES.check_point_path, # 체크포인트 위치 등록한다.
        params={ # 모델 쪽으로 파라메터 전달한다.
            'hidden_size': DEFINES.hidden_size,  # 가중치 크기 설정한다.
            'layer_size': DEFINES.layer_size,  # 멀티 레이어 층 개수를 설정한다.
            'learning_rate': DEFINES.learning_rate,  # 학습율 설정한다.
            'teacher_forcing_rate': DEFINES.teacher_forcing_rate, # 학습시 디코더 인풋 정답 지원율 설정
            'vocabulary_length': vocabulary_length,  # 딕셔너리 크기를 설정한다.
            'embedding_size': DEFINES.embedding_size,  # 임베딩 크기를 설정한다.
            'embedding': DEFINES.embedding,  # 임베딩 사용 유무를 설정한다.
            'multilayer': DEFINES.multilayer,  # 멀티 레이어 사용 유무를 설정한다.
            'attention': DEFINES.attention, #  어텐션 지원 유무를 설정한다.
            'teacher_forcing': DEFINES.teacher_forcing, # 학습시 디코더 인풋 정답 지원 유무 설정한다.
            'loss_mask': DEFINES.loss_mask, # PAD에 대한 마스크를 통한 loss를 제한 한다.
            'serving': DEFINES.serving # 모델 저장 및 serving 유무를 설정한다.
        })


    while 1:



     now = datetime.datetime.now()
     #List
     loadcount = loadcount + 1
     if now.hour == 17 :
         bootcount = 0
         learn()
         break

     dataa = {'id': board_id}
     cookies = {'_ga' : cookie}
     res = requests.post(everytime_list_url, data=dataa,  stream=True, cookies=cookies)
     answer = res.content
     print(answer)

     firstacess = True
     allowPass = False
     root = elemTree.fromstring(answer)

     for child in root.iter("article") :
      postid = int(child.attrib['id'])
      is_mine_comment = False
      if postid <= lastid:
          break

      if firstacess :
          clastid = postid
          firstacess = False

      post = child.attrib['title'] + " " + child.attrib['text']
      print("[글]" + post)
      # with open("./data/chat.log", "a") as myfile:
      #     myfile.write(post + "\n")
      if int(child.attrib['comment']) >= 3 :
         sleep(2)
         cdata = {'id': child.attrib['id']}
         cres = requests.post(everytime_art_url, data=cdata,  stream=True, cookies=cookies)
         best_comment = ""
         bestcount = -1
         croot = elemTree.fromstring(cres.content)
         for comments in croot.iter("comment") :
          print("[댓글]" + comments.attrib['text'])
          if comments.attrib['is_mine'] == "1" :
             is_mine_comment = True
          posvote = int(comments.attrib['posvote'])
          comment_parent_id = comments.attrib['parent_id']
          if bestcount <= posvote and comment_parent_id == "0" :
             best_comment = comments.attrib['text']
             bestcount = posvote
         predic_input_enc, predic_input_enc_length = data.enc_processing([post], char2idx)
         predic_target_dec, _ = data.dec_target_processing([""], char2idx)
         # 예측을 하는 부분이다.
         predictions = classifier.predict(
         input_fn=lambda:data.eval_input_fn(predic_input_enc, predic_target_dec, DEFINES.batch_size))
         # 예측한 값을 인지 할 수 있도록
         # 텍스트로 변경하는 부분이다.

         postfeel = 0

         post = post.replace(","," ")
         best_comment = best_comment.replace(","," ")

         if "씨발" in post or "시발"in post or "ㅅㅂ" in post or "새끼" in post or "ㅅㄲ" in post or "ㅠ"in post or "찐따"in post or "ㅜ" in post :
             postfeel = 1

         if "파이팅" in post or "ㅎㅎ"in post or "ㅋㅋ" in post or "사랑" in post or "좋은" in post :
             postfeel = 2

         if is_mine_comment :
             break

         # with open("./data_in_candidate/ChatBotData.csv", "a") as myfile:
         #     myfile.write(post  + "," + best_comment + "," + str(postfeel) + "\n")
         # comment_write = data.pred2string(predictions, idx2char)
         #
         #
         # if comment_write != "" :
         #     print("[댓글 작성] " + comment_write)
         #     bot.send_message(chat_id = -116418298, text= "[에타 글] 글 : " + post + "\n\n" + "봇의 댓글 : " + comment_write)
         #     cwdata = {'id': child.attrib['id'], 'text' : comment_write, 'is_anonym' : "1"}
         #     cwres = requests.post(everytime_commentw_url, data=cwdata,  stream=True, cookies=cookies)

     lastid = clastid
     with open("everytime.txt", "r+") as f:
         f.seek(0)
         f.write(str(lastid))
     #learnquick()
     start_time= time.time()
     learn()
     end_time= time.time()
     diff_time = end_time - start_time
     if end_time - start_time < 600 :
         sleep_time = 600 - diff_time
         sleep(sleep_time)
     break
 except :
       bot.send_message(chat_id = -116418298, text="[에타 모듈 오류] 크롤링 모듈이 동작하던 중 문제가 발생했습니다. 점검이 필요합니다.")
       print(traceback.format_exc())
       sleep(6)


if __name__ == '__main__':
    main()
