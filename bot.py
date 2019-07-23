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
import model as ml
import tensorflow as tf
from earlystop import EarlyStopping
import random
import math
import os, sys
import data
import datetime
from configs import DEFINES



update_id = 0


def __del__(self):
    bot = telegram.Bot('auth')
    bot.send_message(chat_id = -116418298, text="Penta 서비스가 종료되었습니다.")

def main():
    """Run the bot."""
    global update_id


    # Telegram Bot Authorization Token
    bot = telegram.Bot('auth')
    URL = "https://unopenedbox.com/develop/square/api.php"
    last_message = ""
    bootcount = 0
    lcount = 0
    readingold = False
    readingold_lastcount = 0
    now = datetime.datetime.now()
    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.

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
     sleep(3)
     now = datetime.datetime.now()
     bootcount = bootcount + 1
     lcount = lcount + 1



     try:
         #data = {'a': 'penta_check', 'auth': 'a1s2d3f4g5h6j7k8l9', 'start_num' : '0', 'number' : '15'}
         #res = requests.post(URL, data=data)
         #answer = "[보고]" + res.json()[0]['description'];
         answer = ""

         if bootcount == 1 :
             #answer = "다시 시작했습니다. Penta 버전 1.0.625 밀린 채팅을 읽는 중 입니다..."
             readingold = True
             readingold_lastcount = bootcount
         if readingold_lastcount < bootcount and readingold is True :
             readingold = False
             #bot.send_message(chat_id = -116418298, text="이전글 읽기 완료.")
         if last_message != answer and answer != "" :
            bot.send_message(chat_id = -116418298, text=answer)
            last_message = answer
         if last_message == answer :
             tlm = ""
             last_user = 0
             last_talk = ""
             updates = bot.get_updates(offset=update_id)
             for i in updates:
               if i.message:
                if last_user != i.message.from_user.id :
                    last_talk = tlm
                    tlm = ""
                    last_user = i.message.from_user.id
                    # with open("./data_in/ChatBotData.csv", "a") as myfile:
                    #   myfile.write("\n")

                if i.message.text is not None and tlm != "" :
                 tlm = tlm + " " + i.message.text
                # with open("./data_in/ChatBotData.csv", "a") as myfile:
                #    myfile.write(" " + i.message.text)
                if i.message.text is not None and tlm == "" :
                 tlm = i.message.text
                 # with open("./data_in/ChatBotData.csv", "a") as myfile:
                 #  myfile.write(i.message.text)
                update_id = i.update_id + 1
                now_last_id = updates[-1].update_id


                if tlm != "" and tlm is not None and now_last_id  + 1 <= update_id:
                 readingold_lastcount = readingold_lastcount +1
                 lcount = 0
                 if not readingold :

                  predic_input_enc, predic_input_enc_length = data.enc_processing([tlm], char2idx)
                  predic_target_dec, _ = data.dec_target_processing([""], char2idx)
                  # 예측을 하는 부분이다.
                  predictions = classifier.predict(
                  input_fn=lambda:data.eval_input_fn(predic_input_enc, predic_target_dec, DEFINES.batch_size))
                  # 예측한 값을 인지 할 수 있도록
                  # 텍스트로 변경하는 부분이다.

                  aimessage = data.pred2string(predictions, idx2char)
                  if aimessage != "" :
                   bot.send_message(chat_id = -116418298, text=aimessage)
     except IndexError:
        update_id = None
     except:
         if last_message !=  traceback.format_exc() and "Message text is empty" not in traceback.format_exc():

          bot.send_message(chat_id = -11641828, text="[오류] 오류가 발생했습니다. 점검이 필요합니다. \n"+ traceback.format_exc())
          last_message =  traceback.format_exc()

     logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def echo(bot):
    """Echo the message the user sent."""
    global update_id
    # Request updates after the last update_id
    for update in bot.get_updates(offset=update_id, timeout=10):
        update_id = update.update_id + 1

        if update.message:  # your bot can receive updates without messages
            # Reply to the message
            update.message.reply_text("HI")

if __name__ == '__main__':
    main()
