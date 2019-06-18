```python
from selenium import webdriver
import time
import os
import requests
from selenium.webdriver.common.action_chains import ActionChains
import logging


dir_path=r"c:/users/yelei/desktop/Asian_Stars"
country_en="China"
dir_path_with_country=os.path.join(dir_path,country_en)


def LOGER(LogLevName):
    logger = logging.getLogger('Getting Asian stars logger')
    logger.setLevel(logging.DEBUG)

    logger_filename = os.path.join(dir_path, '{0}_{1}.log'.format(country_en,LogLevName))
    hander = logging.FileHandler(filename=logger_filename, mode='a')
    hander.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s-%(levelname)s:%(message)s')
    hander.setFormatter(formatter)
    logger.addHandler(hander)
    return logger


class Huaban():
    index=0

    def Url_LOG(self):
        return  LOGER('Url_Dir')



    #获取图片url并存到列表urls_list
    def get_picture_url(self, stars_list,picture_num):
        global path


        url = "http://huabanpro.com/"
        # 使用Chrome浏览器模拟打开网页，但是要把下载的chromedriver.exe放在python的文件路径下,
        # 调试好之后换成PhantomJs,速度应该会快一点
        # driver = webdriver.PhantomJs()
        # 下拉滑动浏览器屏幕，具体下拉多少根据自己实际情况决定
        driver = webdriver.PhantomJS(
            executable_path=r'E:\python0907\venv1\Scripts\phantomjs-2.1.1-windows\bin\phantomjs.exe')
        # driver = webdriver.Chrome()
        # 设置全屏
        driver.maximize_window()
        driver.get(url)
        # time.sleep(8)

        # # 点击登录、呼起登录窗口
        driver.find_elements_by_xpath('//a[@class="login bounce btn wbtn"]')[0].click()  #login btn wbtn
        # sign in the username
        try:
            driver.find_elements_by_xpath('//input[@name="email"]')[0].send_keys('13282196367')
            print('user success!')
        except:
            print('user error!')
        time.sleep(3)
        # sign in the pasword
        try:
            driver.find_elements_by_xpath('//input[@name="password"]')[0].send_keys('xiaoye1014')
            print('pw success!')
        except:
            print('pw error!')
        time.sleep(3)
        # click to login
        try:
            driver.find_elements_by_xpath('//a[@class="btn btn18 rbtn"]')[0].click()
            print('click success!')
        except:
            print('click error!')
        time.sleep(1)
        # click to login
        # 搜索图片
        for content in stars_list:
            Huaban.index=0
            path = os.path.join(dir_path_with_country,content)
            # 保存图片到磁盘文件夹 file_path中，默认为当前脚本运行目录下的文件夹
            if not os.path.exists(path):
                os.makedirs(path)
            search_content=driver.find_elements_by_xpath('//input[@placeholder="搜索你喜欢的"]')[0]
            search_content.clear()   ##这个地方很重要
            search_content.send_keys(content)

            # ActionChains(driver).click(driver.find_elements_by_xpath("//form[@id='search_form']/a")[0]).perform()
            driver.find_elements_by_xpath('//form[@id="search_form"]/a')[0].click()
            time.sleep(1)
            i = 0
            page = 1
            global name
            global store_path
            global urls_list
            urls_list = []
            # 获取图片的总数
            # a=driver.find_elements_by_xpath('//a[@class="selected"]/i')[0].text

            if (int)(driver.find_elements_by_xpath('//a[@class="selected"]/i')[0].text)==0:
                print("【%s】搜不到图片！" % content)
                continue
            else:
                pictures_count = driver.find_elements_by_xpath('//a[@class="selected"]/i')[0].text
            # print(pictures_count)
            pages = int(int(pictures_count) / 20)
            # print(pages)
            # 匹配到图片url所在的元素
            url_elements = driver.find_elements_by_xpath('//span[@class="stop"]/../img')
            # 遍历图片元素的列表获取图片的url
            for url_element in url_elements:
                picture_url = url_element.get_attribute("src")[:-3] + "658"  ####
                # 防止获取重复的图片url
                if picture_url not in urls_list:
                    urls_list.append(picture_url)
            print("开始下载【%s】图片："%content)
            while page <= pages:
                while len(urls_list) < 20 * page:
                    driver.execute_script("window.scrollBy(0,1000)")
                    time.sleep(1)
                    url_elements = driver.find_elements_by_xpath('//span[@class="stop"]/../img')
                    for url_element in url_elements:
                        picture_url = url_element.get_attribute("src")[:-3] + "658"
                        if picture_url not in urls_list:
                            urls_list.append(picture_url)
                print("第%s页" % page)

                for download_url in urls_list[20 * (page - 1):20 * page]:
                    i += 1
                    name = content + "_" + str(i)
                    store_path = name + '.jpg'
                    self.store(download_url)
                    if Huaban.index == picture_num:
                        break
                if Huaban.index == picture_num:
                    break
                page += 1
            # 最后一页
            print("第%s页" % int(page))

            if Huaban.index < picture_num:
                while len(urls_list) < int(pictures_count):
                    driver.execute_script("window.scrollBy(0,1000)")
                    time.sleep(1)
                    url_elements = driver.find_elements_by_xpath('//span[@class="stop"]/../img')
                    for url_element in url_elements:
                        picture_url = url_element.get_attribute("src")[:-3] + "658"
                        if picture_url not in urls_list:
                            urls_list.append(picture_url)
                for download_url in urls_list[20 * (page - 1):]:
                    i += 1
                    name = content + "_" + str(i)
                    store_path = name + '.jpg'
                    self.store(download_url)
                    if Huaban.index == picture_num:
                        break
            print("【%s】共%d照片下载成功。"%(content,Huaban.index))

                # 存储图片到本地
    def store(self, picture_url):
        picture = requests.get(picture_url)
        f = open(path + '\\' + store_path, 'wb')
        f.write(picture.content)
        Huaban.index+=1
        print('正在保存图片%s：%s' %(name,picture_url) )
        self.Url_LOG().info("%s 图片url：%s" % (name, picture_url))


if __name__ == "__main__":
    stars_list_dir=os.path.join(dir_path_with_country,'stars_list.txt')
    with open(stars_list_dir) as file:
        _stars_list=file.read().splitlines()
    person_already_down_list = os.listdir(dir_path_with_country)
    stars_list = [file for file in _stars_list if file not in person_already_down_list]

    picture_num=50
    huaban = Huaban()
    huaban.get_picture_url(stars_list,picture_num)
    print('所有图片下载完成！')

```

