import logging
from urllib.parse import urlparse

from tqdm import tqdm

from data_engineering.crawlers.dispatcher import CrawlerDispatcher
from data_engineering.pipelines.ODM.documents import UserDocument

logging.basicConfig(level=logging.INFO)


def crawl_links(user: UserDocument, links: list[str]):
    logging.info(f"Starting to crawl {len(links)} link(s)")
    dispatcher = CrawlerDispatcher()

    # metadata = {}
    successful_crawls = 0
    for link in tqdm(links):
        success, crawled_domain = _crawl_link(dispatcher, user, link)
        successful_crawls += success
        # _add_to_metadata(metadata, crawled_domain, success)

    logging.info(f"Successfully crawled {successful_crawls} / {len(links)} links")
    return links


def _crawl_link(
    dispatcher: CrawlerDispatcher, user: UserDocument, link: str
) -> tuple[bool, str]:
    crawler = dispatcher.get_crawler(link)
    crawler_domain = urlparse(link).netloc

    try:
        crawler.extract(link=link, user=user)
        return (True, crawler_domain)
    except Exception as e:
        logging.warning(f"An error occured while crawling: {e}")
        return (False, crawler_domain)
