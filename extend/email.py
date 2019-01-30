import smtplib
import sys
sys.path.append('..//')
from Config.Config import Config as conf
from email.mime.multipart import MIMEMultipart    
from email.mime.text import MIMEText    
from email.mime.image import MIMEImage 
from email.header import Header 

'''
用法：
在文件中引用（具体路径视情况定）
sys.path.append('..//')
from extend.email import Email as Em

mail=Em()
mail.send()
ok 之后
del mail #关闭通信并删除mail，下次使用时重新定义
'''
class Email(object):
    def __init__(self, *args, **kwargs):
        self.smtp = smtplib.SMTP_SSL('smtp.qq.com',465)
        self.smtp.login(conf.username, conf.password) 
        return super().__init__(*args, **kwargs)
    
    def send(self,
             subj='训练完成！（training done notice）',
             message='Here is LUNA16 Program（Jiabin）!\n now,you Training process is done!\n Go and Start next Jobs.\n                           ---yours.'):
        subject = subj
        msg = MIMEMultipart('mixed') 
        msg['Subject'] = subject
        msg['From'] = 'komo.tan@foxmail.com <komo.tan@foxmail.com>'
        msg['To'] = ";".join(conf.email_list) 
        text = message 
        text_plain = MIMEText(text,'plain', 'utf-8')    
        msg.attach(text_plain) 
        self.smtp.sendmail('komo.tan@foxmail.com', conf.email_list, msg.as_string())
        pass
    def __del__(self):
        self.smtp.quit()
    pass

if __name__=='__main__':
    mail=Email()
    mail.send()
    del mail



