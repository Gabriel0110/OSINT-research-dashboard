'''
email_handler.py
'''

import importlib
import logging
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EmailHandler:
    def __init__(self, sentence_model=None):
        self.email_client = None
        self.mailboxes = []
        self.sentence_model = sentence_model
        self.folder_presets = self.load_folder_presets()
        self.initialize_email_client()

    def is_available(self):
        return self.email_client is not None

    def initialize_email_client(self):
        try:
            win32com = importlib.import_module('win32com.client')
            self.email_client = win32com.Dispatch("Outlook.Application").GetNamespace("MAPI")
            logger.info("Outlook client initialized successfully.")

            self.load_mailboxes()
        except ImportError:
            logger.warning("pywin32 is not installed. Outlook functionality will not be available.")
        except Exception as e:
            logger.error(f"Failed to initialize Outlook client: {e}")

    def load_folder_presets(self):
        try:
            with open('folder_presets.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("No saved folder presets file found. Starting with empty dict.")
            return {}

    def save_folder_presets(self):
        with open('folder_presets.json', 'w') as f:
            json.dump(self.folder_presets, f)
        logger.info(f"Saved {len(self.folder_presets)} folder presets to file.")

    def add_folder_preset(self, preset_name, folders):
        if not folders:
            return False, "No folders specified for preset."

        if not preset_name:
            return False, "Preset name cannot be empty."

        if preset_name in self.folder_presets:
            return False, f"Preset '{preset_name}' already exists."
        
        self.folder_presets[preset_name] = folders
        self.save_folder_presets()

    def remove_folder_preset(self, preset_name):
        if preset_name in self.folder_presets:
            del self.folder_presets[preset_name]
            self.save_folder_presets()

    def get_folder_presets(self):
        logger.info(f"Returning {len(self.folder_presets)} folder presets.")
        return self.folder_presets

    def load_mailboxes(self):
        # try:
        #     with open('mailboxes.json', 'r') as f:
        #         self.mailboxes = json.load(f)
        #     logger.info(f"Loaded {len(self.mailboxes)} mailboxes from file.")
        # except FileNotFoundError:
        #     logger.info("No saved mailboxes file found. Starting with empty list.")
        #     self.mailboxes = []

        logger.info("Loading mailboxes from Outlook...")

        try:
            self.mailboxes = [folder.Name for folder in self.email_client.Folders]
            logger.info(f"Loaded {len(self.mailboxes)} mailboxes from Outlook.")
        except Exception as e:
            logger.error(f"Failed to load mailboxes from Outlook: {e}")
            self.mailboxes = []

    # def save_mailboxes(self):
    #     with open('mailboxes.json', 'w') as f:
    #         json.dump(self.mailboxes, f)
    #     logger.info(f"Saved {len(self.mailboxes)} mailboxes to file.")

    # def add_mailbox(self, mailbox_name):
    #     logger.info(f"Attempting to add mailbox: {mailbox_name}")
    #     if mailbox_name not in self.mailboxes:
    #         try:
    #             # Verify if the mailbox exists
    #             self.email_client.Folders[mailbox_name]
    #             self.mailboxes.append(mailbox_name)
    #             self.save_mailboxes()
    #             logger.info(f"Mailbox '{mailbox_name}' added successfully.")
    #             return True, f"Mailbox '{mailbox_name}' added successfully."
    #         except Exception as e:
    #             logger.error(f"Failed to add mailbox '{mailbox_name}': {str(e)}")
    #             return False, f"Failed to add mailbox '{mailbox_name}': {str(e)}"
    #     else:
    #         logger.warning(f"Mailbox '{mailbox_name}' already exists.")
    #         return False, f"Mailbox '{mailbox_name}' already exists."

    # def remove_mailbox(self, mailbox_name):
    #     if mailbox_name in self.mailboxes:
    #         self.mailboxes.remove(mailbox_name)
    #         self.save_mailboxes()
    #         return True, f"Mailbox '{mailbox_name}' removed successfully."
    #     else:
    #         return False, f"Mailbox '{mailbox_name}' not found."

    def get_available_mailboxes(self):
        return self.mailboxes

    def find_folder(self, root_folder, folder_name):
        if not self.is_available():
            return None
        
        logger.info(f"Searching for folder '{folder_name}' in '{root_folder.Name}'.")

        if root_folder.Name == folder_name:
            logger.info(f"Found folder '{folder_name}'.")
            return root_folder
        for folder in root_folder.Folders:
            logger.info(f"Checking subfolder '{folder.Name}'.")
            found_folder = self.find_folder(folder, folder_name)
            if found_folder:
                logger.info(f"Found folder '{folder_name}'.")
                return found_folder
            
        logger.info(f"Folder '{folder_name}' not found.")
        return None

    def search_outlook(self, query, mailboxes=None, folder_names=None, use_embeddings=False, start_date=None, end_date=None):
        if not self.is_available():
            logger.warning("Email search attempted but Outlook is not available.")
            return []

        results = []

        logger.info(f"Searching Outlook for '{query}'.")
        logger.info(f"Mailboxes: {mailboxes}")
        logger.info(f"Folder names: {folder_names}")
        
        try:
            # if mailboxes is None:
            #     logger.info("No mailboxes specified. Searching default mailbox.")
            #     mailboxes = [self.email_client.GetDefaultFolder(6).Parent]  # Default mailbox
            # else:
            #     mailboxes = [self.email_client.Folders[mailbox] for mailbox in mailboxes if mailbox in self.email_client.Folders]

            for mailbox in mailboxes:
                mailbox = self.email_client.Folders(mailbox)
                logger.info(f"Searching mailbox '{mailbox.Name}'...")

                if folder_names:
                    logger.info(f"Folder names found: {folder_names}")

                    folders = [self.find_folder(mailbox, name) for name in folder_names]
                    logger.info(f"Folders after search: {folders}")

                    folders = [folder for folder in folders if folder is not None]
                    logger.info(f"Folders after filtering: {folders}")
                else:
                    folders = [mailbox.GetDefaultFolder(6)]  # 6 is the Inbox folder
                    logger.info(f"No folder names specified. Searching default folder '{folders[0].Name}'.")

                logger.info(f"Searching {len(folders)} folders in mailbox '{mailbox.Name}'.")

                for folder in folders:
                    results.extend(self.search_outlook_folder(folder, query, use_embeddings, start_date, end_date, self.sentence_model))
        except Exception as e:
            logger.error(f"Error searching Outlook: {e}")

        logger.info(f"Email search completed. Found {len(results)} results.")

        return results

    def search_outlook_folder(self, folder, query, use_embeddings=False, start_date=None, end_date=None, sentence_model=None):
        results = []
        messages = folder.Items
        messages.Sort("[ReceivedTime]", True)

        if use_embeddings and sentence_model:
            query_embedding = sentence_model.encode([query])[0]

        # Convert start_date and end_date to Outlook's date format
        outlook_start_date = start_date.strftime("%m/%d/%Y") if start_date else None
        outlook_end_date = end_date.strftime("%m/%d/%Y") if end_date else None
        
        # Apply date filter
        date_filter = ""
        if outlook_start_date:
            date_filter += f"[ReceivedTime] >= '{outlook_start_date}'"
        if outlook_end_date:
            date_filter += f" AND [ReceivedTime] <= '{outlook_end_date}'" if date_filter else f"[ReceivedTime] <= '{outlook_end_date}'"
        
        if date_filter:
            messages = messages.Restrict(date_filter)
        
        for message in messages:
            try:
                subject = message.Subject
                body = message.Body

                if use_embeddings:
                    text_embedding = self.sentence_model.encode([subject + " " + body])[0]
                    similarity = cosine_similarity([query_embedding], [text_embedding])[0][0]
                    if similarity > 0.5:  # Adjust this threshold as needed
                        results.append({
                            "title": subject,
                            "snippet": body[:200] + "...",
                            "source": "Outlook Email",
                            "published": message.ReceivedTime.strftime("%a, %d %b %Y %H:%M:%S %z"),
                        })
                else:
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