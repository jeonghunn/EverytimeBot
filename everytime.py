"""Simple Bot to reply to Telegram messages.
This is built on the API wrapper, see echobot2.py to see the same example built
on the telegram.ext bot framework.
This program is dedicated to the public domain under the CC0 license.
"""
import logging
import telegram
import requests, json
import traceback
from chatbot import ChatBotT
from config import FLAGS
from time import sleep
from dialog import Dialog
import tensorflow as tf
from model import Seq2Seq
from earlystop import EarlyStopping
import random
import math
import os, sys
import datetime
import xml.etree.ElementTree as elemTree




def trainbot(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_candidate_dir)
        beq = False
        bot.send_message(chat_id = -116418298, text="공부 상황을 보고 싶다면 '로그'라 명령 해주세요.")

        if  ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            bot.send_message(chat_id = -116418298, text="파일에서 모델을 읽는 중 입니다..")
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            bot.send_message(chat_id = -116418298, text="새로운 모델을 생성하는 중 입니다.")
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        last_message = 0
        update_id = 0

        total_batch = int(math.ceil(len(dialog.examples)/float(batch_size)))
        early_stopping = EarlyStopping(patience=150, verbose=1)

        for step in range(total_batch * epoch):

            enc_input, dec_input, targets = dialog.next_batch(batch_size)

            _, loss = model.train(sess, enc_input, dec_input, targets)

            if (step + 1) % 100 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets)

                if(beq) :
                    bot.send_message(chat_id = -116418298, text="공부하는 중...Step : " + '%06d' % model.global_step.eval() + " 오차 : " + '{:.6f}'.format(loss))

                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))
            if early_stopping.validate(loss):
                bot.send_message(chat_id = -116418298, text="더 이상 공부할 필요가 없다 판단하여 이번 회차 공부를 종료합니다.")
                break
            for update in bot.get_updates(offset=update_id):
                if update.message is not None :
                    if update.message.text == "음소거" :
                        beq = False
                    if update.message.text == "로그" :
                        last_message = 0
                        update_id = 0
                        beq = True
                update_id = update.update_id + 1
            # if update_id != last_message and last_message != 0 :
            #     print("update : "  + str(update_id) + "last: " + str(last_message))
            #     break
            last_message = update_id
        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        checkpoint_path = os.path.join(FLAGS.train_candidate_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        os.system("cp ./data/chatc.voc ./data/chat.voc")
    bot.send_message(chat_id = -116418298, text="공부한 것에 대한 최적화 완료.")

def learn():
    try:
        bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')
        bot.send_message(chat_id = -116418298, text="[학습 프로세스 시작] 새로운 변경 내역을 학습합니다. 이 작업은 하루에 한번정도 실행됩니다.")
        bot.send_message(chat_id = -116418298, text="공부할 새로운 단어 사전을 만드는 중입니다...")
        dialog = Dialog()
        dialog.build_vocab(FLAGS.data_path, FLAGS.voc_candidate_path)
        bot.send_message(chat_id = -116418298, text="메모리 한계를 초과하는 것을 막기 위해 공부할 내용을 분권 중입니다...")
        lines_per_file = 4000
        smallfile = None
        index = 0
        with open('./data/chat.log') as bigfile:
            for lineno, line in enumerate(bigfile):

                if lineno % lines_per_file == 0:
                    if smallfile:
                        smallfile.close()
                    small_filename = 'chat{}.log'.format(lineno + lines_per_file)
                    smallfile = open(small_filename, "w")
                    index = index + 1
                    bot.send_message(chat_id = -116418298, text=str(index) + "번째 책 만드는 중...")
                smallfile.write(line)
        if smallfile:
            smallfile.close()
        bot.send_message(chat_id = -116418298, text=str(index) + "개의 파일로 분리되었습니다.")
        bot.send_message(chat_id = -116418298, text="단어장을 불러오는 중입니다.")
        dialog.load_vocab(FLAGS.voc_path)
        bot.send_message(chat_id = -116418298, text="학습을 시작합니다... 새로운 모델을 생성하는 중 입니다.")
        folder = './modelc'
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
                bot.send_message(chat_id = -116418298, text="[오류] 기존 모델을 삭제하는데 실패했습니다.")

        with tf.Session() as sess:
            print("새로운 모델을 생성하는 중 입니다.")

            sess.run(tf.global_variables_initializer())
        for i in range(0, index):
            ii = i+1
            tf.reset_default_graph()
            bot.send_message(chat_id = -116418298, text=str(ii) +"번째 책에 대한 공부 진행 중...")
            print(str(ii) +"번째 책에 대한 공부 진행 중...")
            dialog = Dialog()
            dialog.load_vocab(FLAGS.voc_candidate_path)
            dialog.load_examples("./chat" + str(ii) + "000.log")
            trainbot(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
        bot.send_message(chat_id = -116418298, text="오늘 할 공부가 다 끝났습니다. 적용을 위해 봇을 재시작합니다.")
        bootcount = 0
        os.system("nohup ./restart.sh")
        sleep(0.2) # 200ms to CTR+C twice
    except:
        bot.send_message(chat_id = -116418298, text="[오류] 학습을 진행하던 도중 문제가 발생하여 종료하였습니다. 점검이 필요합니다.\n" + traceback.format_exc())
        print(traceback.format_exc())

def learnquick():
    try:
        bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')
        lines_per_file = 4000
        smallfile = None
        dialog = Dialog()
        index = 0
        with open('./data/chat.log') as bigfile:
            for lineno, line in enumerate(bigfile):

                if lineno % lines_per_file == 0:
                    if smallfile:
                        smallfile.close()
                    small_filename = 'chat{}.log'.format(lineno + lines_per_file)
                    smallfile = open(small_filename, "w")
                    index = index + 1
                smallfile.write(line)
        if smallfile:
            smallfile.close()
        dialog.load_vocab(FLAGS.voc_candidate_path)

        for i in range(0, index):
            ii = i+1
            tf.reset_default_graph()
            print(str(ii) +"번째 책에 대한 공부 진행 중...")
            dialog = Dialog()
            dialog.load_vocab(FLAGS.voc_candidate_path)
            dialog.load_examples("./chat" + str(ii) + "000.log")
            trainbot(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)


        os.system("nohup ./restart.sh")
        sleep(0.2) # 200ms to CTR+C twice
    except:
        bot.send_message(chat_id = -116418298, text="[오류] 학습을 진행하던 도중 문제가 발생하여 종료하였습니다. 점검이 필요합니다.\n" + traceback.format_exc())
        print(traceback.format_exc())





def main():
    """Run the bot."""

global update_id
bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')

while 1:
 try:
    lastid = 0
    with open("everytime.txt", "r+") as f:
        lastid = int(f.seek(0)) # read everything in the file

    clastid =0
    loadcount = 0
    everytime_list_url = "http://everytime.kr/find/board/article/list"
    everytime_art_url = "https://everytime.kr/find/board/comment/list"
    everytime_commentw_url = "https://everytime.kr/save/board/comment"
    cookie = "GA1.2.1817913000.1559891507; _gid=GA1.2.692934811.1561434809; _gat_gtag_UA_22022140_4=1; sheet_visible=1; etsid=s%3A31YN_tXVNUNIn7FHotFxJQmXnq9cUJhJ.%2Byez3XEphFIMzgq0h4OVx7A%2F%2FI%2BJYpmG7DIwylTCq2A"
    board_id = "374911"

    while 1:

     tf.reset_default_graph()
     try :
      chatbot = ChatBotT(FLAGS.voc_path, FLAGS.train_dir)
     except :
      bot.send_message(chat_id = -116418298, text="[오류] 학습 파일에 문제가 있어 재학습해야 합니다.")
      learn()

     tf.reset_default_graph()
     chatbot = ChatBotT(FLAGS.voc_path, FLAGS.train_dir)
     now = datetime.datetime.now()
     #List
     loadcount = loadcount + 1
     if now.hour == 17 :
         bootcount = 0
         learn()
         break

     data = {'id': board_id}
     cookies = {'_ga' : cookie}
     res = requests.post(everytime_list_url, data=data,  stream=True, cookies=cookies)
     answer = res.content
     print(answer)

     firstacess = True
     allowPass = False
     root = elemTree.fromstring(answer)

     for child in root.iter("article") :
      postid = int(child.attrib['id'])
      if postid <= lastid  and allowPass is False:
          break

      if firstacess :
          clastid = postid
          firstacess = False

      post = child.attrib['title'] + " " + child.attrib['text']
      print("[글]" + post)
      with open("./data/chat.log", "a") as myfile:
          myfile.write(post + "\n")
      if int(child.attrib['comment']) > 4 :
         sleep(2)
         cdata = {'id': child.attrib['id']}
         cres = requests.post(everytime_art_url, data=cdata,  stream=True, cookies=cookies)

         croot = elemTree.fromstring(cres.content)
         for comments in croot.iter("comment") :
             print("[댓글]" + comments.attrib['text'])
             with open("./data/chat.log", "a") as myfile:
                 myfile.write(comments.attrib['text'] + "\n")
                 comment_write = chatbot.run(post)

         if comment_write != "" :
             print("[댓글 작성] " + comment_write)
             cwdata = {'id': child.attrib['id'], 'text' : comment_write, 'is_anonym' : "0"}
             cwres = requests.post(everytime_commentw_url, data=cwdata,  stream=True, cookies=cookies)
         sleep(4)
     lastid = clastid
     with open("everytime.txt", "r+") as f:
         f.seek(0)
         f.write(str(lastid))
     learnquick()
 except :
       bot.send_message(chat_id = -116418298, text="[에타 모듈 오류] 크롤링 모듈이 동작하던 중 문제가 발생했습니다. 점검이 필요합니다.")
       print(traceback.format_exc())
       sleep(6)


if __name__ == '__main__':
    main()