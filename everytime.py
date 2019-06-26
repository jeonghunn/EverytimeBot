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


def main():
    """Run the bot."""

global update_id
bot = telegram.Bot('868556658:AAGi2oU006nQxxy7oVfT0ZkHEGbIXpsYhn8')

while 1:
 try:
    lastid = 0
    with open("everytime.txt", "r+") as f:
        lastid = int(f.seek(0)) # read everything in the file

    tf.reset_default_graph()
    chatbot = ChatBotT(FLAGS.voc_path, FLAGS.train_dir)

    clastid =0
    loadcount = 0
    everytime_list_url = "http://everytime.kr/find/board/article/list"
    everytime_art_url = "https://everytime.kr/find/board/comment/list"
    everytime_commentw_url = "https://everytime.kr/save/board/comment"
    cookie = "GA1.2.1817913000.1559891507; _gid=GA1.2.692934811.1561434809; _gat_gtag_UA_22022140_4=1; sheet_visible=1; etsid=s%3A31YN_tXVNUNIn7FHotFxJQmXnq9cUJhJ.%2Byez3XEphFIMzgq0h4OVx7A%2F%2FI%2BJYpmG7DIwylTCq2A"
    board_id = "374911"

    while 1:
     #List
     loadcount = loadcount + 1
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
         f.write(lastid)
     if loadcount > 3 :
         break
     sleep(600)
 except :
       bot.send_message(chat_id = -116418298, text="[에타 모듈 오류] 크롤링 모듈이 동작하던 중 문제가 발생했습니다. 점검이 필요합니다.")
       print(traceback.format_exc())
       sleep(6)


if __name__ == '__main__':
    main()