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

update_id = 0


def __del__(self):
    bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')
    bot.send_message(chat_id = -116418298, text="Penta 서비스가 종료되었습니다.")


def trainbot(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
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
        early_stopping = EarlyStopping(patience=400, verbose=1)

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
            if update_id != last_message and last_message != 0 :
              print("update : "  + str(update_id) + "last: " + str(last_message))
              break
            last_message = update_id
        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    bot.send_message(chat_id = -116418298, text="공부한 것에 대한 최적화 완료.")

def learn():
    try:
     bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')
     bot.send_message(chat_id = -116418298, text="[학습 프로세스 시작] 새로운 변경 내역을 학습합니다. 이 작업은 하루에 한번정도 실행됩니다.")
     bot.send_message(chat_id = -116418298, text="공부할 새로운 단어 사전을 만드는 중입니다...")
     dialog = Dialog()
     dialog.build_vocab(FLAGS.data_path, FLAGS.voc_path)
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
     folder = './model'
     # for the_file in os.listdir(folder):
     #  file_path = os.path.join(folder, the_file)
     #  try:
     #    if os.path.isfile(file_path):
     #        os.unlink(file_path)
     #    #elif os.path.isdir(file_path): shutil.rmtree(file_path)
     #  except Exception as e:
     #    print(e)
     #    bot.send_message(chat_id = -116418298, text="[오류] 기존 모델을 삭제하는데 실패했습니다.")

    # with tf.Session() as sess:
     # print("새로운 모델을 생성하는 중 입니다.")

     # sess.run(tf.global_variables_initializer())
     for i in range(0, index):
         ii = i+1
         tf.reset_default_graph()
         bot.send_message(chat_id = -116418298, text=str(ii) +"번째 책에 대한 공부 진행 중...")
         print(str(ii) +"번째 책에 대한 공부 진행 중...")
         dialog = Dialog()
         dialog.load_vocab(FLAGS.voc_path)
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
        bot.send_message(chat_id = -116418298, text="채팅방이 조용하니 공부좀 하고 오겠습니다.")
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
        dialog.load_vocab(FLAGS.voc_path)

        for i in range(0, index):
            ii = i+1
            tf.reset_default_graph()
            print(str(ii) +"번째 책에 대한 공부 진행 중...")
            dialog = Dialog()
            dialog.load_vocab(FLAGS.voc_path)
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



    # Telegram Bot Authorization Token
    bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')
    URL = "https://unopenedbox.com/develop/square/api.php"
    last_message = ""
    bootcount = 0
    lcount = 0
    readingold = False
    readingold_lastcount = 0
    now = datetime.datetime.now()
    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.
    if FLAGS.train:
        learn()

    try :
        chatbot = ChatBotT(FLAGS.voc_path, FLAGS.train_dir)
    except :
        bot.send_message(chat_id = -116418298, text="[오류] 학습 파일에 문제가 있어 재학습해야 합니다.")
        folder = './model'
        for the_file in os.listdir(folder):
         file_path = os.path.join(folder, the_file)
         try:
           if os.path.isfile(file_path):
               os.unlink(file_path)
           #elif os.path.isdir(file_path): shutil.rmtree(file_path)
         except Exception as e:
           print(e)
           bot.send_message(chat_id = -116418298, text="[오류] 기존 모델을 삭제하는데 실패했습니다.")
        learn()
    tf.reset_default_graph()
    chatbot = ChatBotT(FLAGS.voc_path, FLAGS.train_dir)
    while 1:
     sleep(10)
     now = datetime.datetime.now()
     bootcount = bootcount + 1
     lcount = lcount + 1
     if now.hour == 17 :
         bootcount = 0
         learn()
     if lcount > 60 :
         lcount = 0
         learnquick()
     try:
         #data = {'a': 'penta_check', 'auth': 'a1s2d3f4g5h6j7k8l9', 'start_num' : '0', 'number' : '15'}
         #res = requests.post(URL, data=data)
         #answer = "[보고]" + res.json()[0]['description'];
         answer = "보고드릴 사항이 없습니다."

         if bootcount == 1 :
             answer = "다시 시작했습니다. Penta 버전 1.0.625 밀린 채팅을 읽는 중 입니다..."
             readingold = True
             readingold_lastcount = bootcount
         if readingold_lastcount < bootcount and readingold is True :
             readingold = False
             bot.send_message(chat_id = -116418298, text=chatbot.run(bot.get_updates()[-1].message.text))
         if last_message != answer :
            bot.send_message(chat_id = -116418298, text=answer)
            last_message = answer
         if last_message == answer :
             tlm = ""
             last_user = 0

             for i in bot.get_updates(offset=update_id):
               if i.message:
                if i.message.text == "머신러닝시작" :
                    learn()
                    break
                if i.message.text == "빠른학습시작" :
                    learnquick()
                    break
                if last_user != i.message.from_user.id :
                    tlm = ""
                    last_user = i.message.from_user.id
                    with open("./data/chat.log", "a") as myfile:
                      myfile.write("\n")

                if i.message.text is not None and tlm != "" :
                 tlm = tlm + " " + i.message.text
                 with open("./data/chat.log", "a") as myfile:
                  myfile.write(" " + i.message.text)
                if i.message.text is not None and tlm == "" :
                 tlm = i.message.text
                 with open("./data/chat.log", "a") as myfile:
                  myfile.write(i.message.text)
                update_id = i.update_id + 1



                if tlm != "" and tlm is not None:
                 readingold_lastcount = readingold_lastcount +1
                 lcount = 0
                 if not readingold :
                   bot.send_message(chat_id = -116418298, text=chatbot.run(tlm))
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