import os
import time
import yagmail
from loguru import logger


def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory1 = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_memory2 = int(gpu_status[6].split('/')[0].split('M')[0].strip())
    gpu_memory3 = int(gpu_status[10].split('/')[0].split('M')[0].strip())
    gpu_memory4 = int(gpu_status[14].split('/')[0].split('M')[0].strip( ))
    # 四卡
    return gpu_memory1, gpu_memory2, gpu_memory3, gpu_memory4


while 1:
    print('current time: ', time.asctime())
    tmp = gpu_info()
    # for i in range(0, len(tmp)):
    #     print("gpu[%d] memory usage: %d\n" % (i, tmp[i]))
    if min(tmp) < 2000:
        index = tmp.index(min(tmp))
        yagmail_server = yagmail.SMTP(user="2687336030@qq.com", password="bimmjrfvnnfpdeaj", host="smtp.qq.com")
        # 密码填入生成的授权码
        email_name = ["2687336030@qq.com"]
        email_title = ["Congratulations !"]
        email_content = ["got gpu %d available !" % index]
        yagmail_server.send(to=email_name, subject=email_title, contents=email_content)
        yagmail_server.close()
        logger.info("邮件发送完毕！")
        break
    time.sleep(60)



#
