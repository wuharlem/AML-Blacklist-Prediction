import requests
import re
import tqdm
import time

from bs4 import BeautifulSoup


class Crawler:
    def __init__(self):
        
        # Default user agent, unless instructed by the user to change it.
        self.USER_AGENT = 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0)'
        
        self.headers = {'User-Agent': self.USER_AGENT,}
        
        self.class_name = {'news.cnyes.com': "_2E8y", 
                           'www.chinatimes.com': 'article-body',
                           'm.ctee.com.tw':'entry-content',
                           'udn.com': "article-content__paragraph",
                           'news.ltn.com.tw': 'text boxTitle boxText',
                           'news.mingpao.com': 'txt4',
                           'technews.tw': 'indent',
                           'www.businesstoday.com.tw': 'cke_editable font__select-content',
                           'www.ettoday.net': {'amp': 'content-container', 'news':'story'},
                           'www.hk01.com': 'sc-bwzfXH liBCIH sc-bdVaJa iMCZeY',
                           'sina.com.hk': 'news-body',
                           'www.bnext.com.tw': 'main_content',
                           'finance.technews.tw': 'indent',
                           'www.cw.com.tw': 'nevin',
                           'www.mirrormedia.mg': 'article_main',
                           'hk.on.cc': 'breakingNewsContent',
                           'www.wealth.com.tw': 'entry-content cms-box pt30 mb50',
                           'tw.news.yahoo.com': 'articleBody',
                           'www.coolloud.org.tw': 'field-item even',
                           'mops.twse.com.tw': 'zoom',
                           'www.setn.com': 'Content1',
                           'www.managertoday.com.tw': 'articleBody',
                           'www.hbrtaiwan.com': 'content-area--article artic-content column mainArticle', 
                           'm.ltn.com.tw': 'articleBody',
                           'ccc.technews.tw': 'indent',
                           'money.udn.com': 'article_body', 
                           'ec.ltn.com.tw': 'text',
                           'www.cna.com.tw': 'paragraph', 
                          }
        
        self.manual_domain = ['www.fsc.gov.tw','ol.mingpao.com',\
                              'www.nextmag.com.tw','ent.ltn.com.tw','news.ebc.net.tw','estate.ltn.com.tw',\
                              'www.nownews.com','www.storm.mg','house.ettoday.net']
        
        self.replace_word = ['\n', '\r', '\u3000', '\xa0', '\br']
        
        
    def search(self, dom, page, link, verbose=False):
        '''
        Using html response to get the wanted article.
        :param bs4 page: html response.
        :return: article string.
        :raises bs4 Error: An exception is raised on error.
        '''
        if dom in self.manual_domain:
            return '手動找'
        
        try:
            content = self.add_title(dom, page)
            
            ## Special Case: 沒有用 <p> 的網站
            if dom == 'www.mirrormedia.mg' or dom == 'hk.on.cc' or dom == 'mops.twse.com.tw':
                for line in self.get_lines(dom, page, link):
                    try:
                        content+=line.text
                        if '更新時間' in line.text:
                            break
                    ## text not in line
                    except:
                        pass

            ## Special Case 家事事件公告
            elif dom == 'domestic.judicial.gov.tw':
                content = page.find('pre').text
                
            ## Special Case TVBS and 新浪
            elif dom == 'news.tvbs.com.tw':
                # kill all script and style elements
                for script in page(["script", "style"]):
                    script.decompose()    # rip it out
                    
                # get text
                if dom == 'sina.com.hk':
                    text = page.get_text()
                else:
                    text = page.find('div', class_='h7 margin_b20').get_text()
                # break into lines and remove leading and trailing space on each
                lines = (line.strip() for line in text.splitlines())
                # break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                # drop blank lines
                text = '\n'.join(chunk for chunk in list(chunks)[:-1] if chunk)
                content = text
                        

            ## General Case
            else:
                for line in self.get_lines(dom, page, link):
                    content+=line.text 
            
            ## Delete some words
            for c in self.replace_word:
                content = content.replace(c, '')
            
            return content

        ## 以上pattern都找不到
        except Exception as e:
            if verbose:
                print(e, link)
            return f'文章已被刪除 404 or 例外'
    
    def add_title(self, dom, page):
        '''
        If content has a title or not
        '''
    
        ## Add Title Case 1: 鉅亨新聞
        if dom == 'news.cnyes.com':
            title = ''
            for line in page.find("div", class_='_uo1n'):
                try:
                    title+=line.text
                ## text not in line
                except:
                    pass
            content = title

        ## Add Title Case 2: 數位時代
        elif dom == 'www.bnext.com.tw':
            title = ''
            try:
                title = page.find('div', class_='article_summary').text
            ## text not in page
            except:
                pass
            content = title    
            
        elif dom == 'www.hbrtaiwan.com':
            title = ''
            try:
                title = page.find('blockquote').text
            except:
                pass
            content = title

        else:
            content = ""
        
        return content
                
    def get_lines(self, dom, page, link):
        '''
        Select the html element from the bs4 html response.
        '''

        div = page.find("div", class_=self.class_name[dom])
        article = page.find("article", class_=self.class_name[dom])
        
        ## Case 1 div
        if dom == 'udn.com':
            lines = div.find('section').find_all('p')
        elif dom == 'news.ltn.com.tw':
            lines = div.find_all('p')[1:-1]
        elif dom == 'sina.com.hk':
            lines = div.find_all('p')
        elif dom == 'hk.on.cc':
            lines = div
        elif dom == 'www.coolloud.org.tw':
            lines = page.find_all("div", class_=self.class_name[dom])[3:]
        elif dom == 'www.ettoday.net':
            if 'amp_news' in link:
                lines = page.find("div", class_=self.class_name[dom]['amp']).find_all('p')
            else:
                lines = page.find("div", class_=self.class_name[dom]['news']).find_all('p')
        
        ## Case 2 article
        elif dom == 'news.mingpao.com':
            lines = article.find_all('p')[:-1]
        elif dom == 'www.bnext.com.tw':
            lines = article.find_all('p')
        elif dom == 'www.hk01.com':
            lines = article
        elif dom == 'tw.news.yahoo.com':
            lines = page.find('article', itemprop = self.class_name[dom]).find_all('p')
    
        ## others
        elif dom == 'www.cw.com.tw':
            lines = page.find("section", class_=self.class_name[dom]).find_all('p')
        elif dom == 'www.mirrormedia.mg':
            lines = page.find("main", class_=self.class_name[dom])
        elif dom == 'mops.twse.com.tw':
            lines = page.find('div', {'id': self.class_name[dom]})
        elif dom == 'money.udn.com':
            lines = page.find('div', {'id': self.class_name[dom]}).find_all('p')
            if len(lines) <= 1:
                return ['a']
        elif dom == 'www.setn.com':
            lines = page.find('div', {'id': self.class_name[dom]}).find_all('p')
        elif dom == 'www.managertoday.com.tw':
            lines = page.find_all('p')
        elif dom == 'm.ltn.com.tw':
            lines = page.find('div', itemprop = self.class_name[dom]).find_all('p')
            
        ## General case
        else:
            lines = div.find_all('p')
            
        return lines
            
    
    def get_page(self, link):
        '''
        Get html response 
        '''
        page = requests.get(link, headers = self.headers)
        
        if 'domestic.judicial.gov.tw' in link:
            page.encoding = 'big5' # parsing要換
        
        return page
    
    def crawl(self, link, verbose=False):
        '''
        Execute crawling process
        '''
        dom  = link.split('//')[1].split('/')[0]
        
        try:
            page = self.get_page(link)
            bs4 = BeautifulSoup(page.content, 'html.parser')
            return self.search(dom, bs4, link, verbose)
        
        except Exception as e:
            print(e)
            print('blocked crawling')
            
    def crawling_process(self, df, test=False, testing_num=100, verbose=False):
        '''
        API to start crawling process and save to the df.
        :param bool test: If it's a test process.
        :param int testing_num: The number of data to test.
        '''
        for i, link in enumerate(tqdm.tqdm(df['hyperlink'].values)):
            if i > testing_num and test:
                break
            df.loc[i, 'article'] = self.crawl(link, verbose)
            
    def crawling_by_domain_process(self, df, domain, verbose = False):
        '''
        API to start crawling process by domain name and save to the df.
        '''
        df_domain = self.get_domain_dataframe(df, domain)
        index = list(df_domain.index)
        
        for i, link in zip(index, tqdm.tqdm(df_domain['hyperlink'].values)):
            df.loc[i, 'article'] = self.crawl(link, verbose)
            
    def get_domain_dataframe(self, df, domain):
        '''
        API to return the dataframe with specific domain.
        '''
        return df[df.domain == domain]