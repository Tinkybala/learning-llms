import logging
from urllib.parse import urlparse

from data_engineering.crawlers.base_crawler import BaseCrawler
from data_engineering.crawlers.custom import CustomArticleCrawler
from data_engineering.crawlers.github import GithubCrawler

logging.basicConfig(level=logging.INFO)


class CrawlerDispatcher:
    def __init__(self):
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        dispatcher = cls()
        return dispatcher

    # def register_medium(self) -> "CrawlerDispatcher":
    #     self.register("https://medium.com", MediumCrawler)
    #     return self

    # def register_linkedin(self) -> "CrawlerDispatcher":
    #     self.register("https://linkedin.com", LinkedInCrawler)
    #     return self

    def register_github(self) -> "CrawlerDispatcher":
        self.register("https://github.com", GithubCrawler)

        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc  # example: "linkedin.com"
        self._crawlers[domain] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for netloc, crawler in self._crawlers:
            if netloc in str:
                return crawler
        
        # If no registered crawlers available for url
        logging.warning(
            f"No crawler found for {url}. Defaulting to CustomArticleCrawler."
        )
        return CustomArticleCrawler()
