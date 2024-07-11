import importlib
import logging

logger = logging.getLogger(__name__)

class EmailHandler:
    def __init__(self):
        self.email_client = None
        self.initialize_email_client()

    def initialize_email_client(self):
        try:
            win32com = importlib.import_module('win32com.client')
            self.email_client = win32com.Dispatch("Outlook.Application").GetNamespace("MAPI")
            logger.info("Outlook client initialized successfully.")
        except ImportError:
            logger.warning("pywin32 is not installed. Outlook functionality will not be available.")
        except Exception as e:
            logger.error(f"Failed to initialize Outlook client: {e}")

    def is_available(self):
        return self.email_client is not None

    def get_available_mailboxes(self):
        if not self.is_available():
            return []
        return [folder.Name for folder in self.email_client.Folders]

    def find_folder(self, root_folder, folder_name):
        if not self.is_available():
            return None
        if root_folder.Name == folder_name:
            return root_folder
        for folder in root_folder.Folders:
            found_folder = self.find_folder(folder, folder_name)
            if found_folder:
                return found_folder
        return None

    def search_outlook(self, query, mailboxes=None, folder_names=None, use_embeddings=False):
        if not self.is_available():
            logger.warning("Email search attempted but Outlook is not available.")
            return []

        results = []
        
        if mailboxes is None:
            mailboxes = [self.email_client.GetDefaultFolder(6).Parent]  # Default mailbox
        else:
            mailboxes = [self.email_client.Folders[mailbox] for mailbox in mailboxes if mailbox in self.email_client.Folders]

        for mailbox in mailboxes:
            if folder_names:
                folders = [self.find_folder(mailbox, name) for name in folder_names]
                folders = [folder for folder in folders if folder is not None]
            else:
                folders = [mailbox.GetDefaultFolder(6)]  # 6 is the Inbox folder

            for folder in folders:
                results.extend(self.search_outlook_folder(folder, query, use_embeddings))

        return results

    def search_outlook_folder(self, folder, query, use_embeddings=False):
        results = []
        messages = folder.Items
        messages.Sort("[ReceivedTime]", True)
        
        for message in messages:
            try:
                subject = message.Subject
                body = message.Body
                
                if query.lower() in subject.lower() or query.lower() in body.lower():
                    results.append({
                        "title": subject,
                        "snippet": body[:200] + "...",
                        "source": "Outlook Email",
                        "published": message.ReceivedTime.strftime("%a, %d %b %Y %H:%M:%S %z"),
                    })
            except Exception as e:
                logger.error(f"Error processing email: {e}")

        return results