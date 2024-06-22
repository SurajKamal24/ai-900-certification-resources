# Udemy: AI-900 Microsoft Azure AI Fundamentals in a Weekend

- **Section 1: AI 900 - Microsoft Azure AI Fundamentals: Getting started**
    - **AI 900 - Azure AI Fundamentals - Introduction**
        1. Azure has 200+ services. Exam expects you to understand 20+ services related to AI. Machine learning, Text analytics, Cognitive services, and Speech
        2. Exam tests your AI fundamentals and decision making abilities
- **Section 2: AI 900 - Microsoft Azure AI Fundamentals: Introduction to AI and ML**
    - **Step 01: Introduction to Artificial intelligence & Machine learning**
        
        Artificial intelligence
        
        1. Self-driving cars
        2. Spam filters
        3. Email classification
        4. Fraud detection
        
        What is AI?
        
        The theory and development of computer systems able to perform tasks normally requiring human intelligence, such as visual perception, speech recognition, decision making, and translation between languages
        
        Understanding types of AI
        
        1. Strong artificial intelligence (or general AI): Intelligence of machine = Intelligence of human
            1. A machine that can solve problems, learn, and plan for the future
            2. An expert at everything (including learning to play all sports and games!)
            3. Learns like child, building on it’s own experiences
            4. We are far away from acheiving this! (Estimates: few decades to never)
        2. Narrow AI (or weak AI): Focuses on specific task
            1. Examples: Self-driving cars and virtual assistants
            2. Machine learning: Learn from data (examples)
    - **Step 02: Exploring Machine learning examples**
        
        Identifying objects from images
        
        1. Computer vision service in Azure - Analyze and describe images, Read text in imagery (OCR), Read handwriting in imagery, Recogize celebrities and landmarks
        2. Alpha go - Reinforcement learning 
    - **Step 03: Exploring Machine learning vs Traditional programming**
        
        Traditional programming: Based on rules
        
        1. IF this DO that
        2. Example: Predict price of a home. Design an algorithm taking all factors into consideration: Location, Home size, Age, Condition, Market, Economy etc
        
        Machine learning: Learning from examples (Not rules)
        
        1. Give millions of examples
        2. Create a model
        3. Use the model to make predictions
        
        Challenges
        
        1. No of examples needed
        2. Availability of skilled personnel
        3. Complexity in implementing MLOps
        
        Three approaches to building AI solutions in Azure
        
        1. Use Pre-Trained models: Azure cognitive services. Get intelligence from text, images, audio, and video
        2. Build simple models: Without needing data scientists. Limited/no-code experience, Example: Custom vision, Example: Azure machine learning - Automated machine learning
        3. Build complex models: Using data scientists and team. Build your own ML models from zero (code-experienced). Example: Using Azure machine learning
        
        Use AI with caution
        
        1. Challenges, risks, and principles
    - **Step 04: Machine learning fundamentals - Scenarios**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Categorize: Building a computer system as intelligent as a human. An expert at everything (all sports and games) | Strong AI |
        | Categorize: Building a computer system that focuses on specific task (self-driving cars, virtual assistants, object detection from images) | Narrow AI or (weak AI) |
        | Category of AI that focuses on learning from data (examples) | Machine learning |
        | How is ML different from traditional programming? | Traditional programming: Rules Machine learning: Examples |
        | Which azure service helps you use Pre-trained models? | Azure cogintive services |
        | Which azuer service helps you build simple models without needing data scientists or AI/ML skills? | Azure machine learning (Automated machine learning), Custom vision |
        | Which azure service helps you build complex ML models? | Azure machine learning |
    - **Quiz 1: Section quiz**
        1. Categorize: You are building a system to detect specific objects in images - **Weak AI**
        2. Most of the AI and ML solutions we work with today are examples of: - **Weak AI**
        3. True or False: Machine learning is a category of AI that focuses on learning from data (examples). - **True**
        4. Which Azure service helps you build complex ML models? - **Azure machine learning**
    - **Step 05: Exploring different types of AI workloads**
        
        
        | Example | AI workload type |
        | --- | --- |
        | Filtering inappropriate content on social media; Recommending products based on user history; Adjusting website content based on user preferences | Content moderation & Personalization |
        | Facial recognition systems; Self-driving car navigation systems; Object detection in surveillance videos; Augmented reality applications | Computer vision workloads |
        | Language translation services; Voice recognition and response systems; Sentiment analysis in customer feedback | Natural language processing workloads |
        | Analyzing large datasets to uncover trends; Extracting useful information from unstructured data; Mining customer data for insights; Predict analyics in business intelligence | Knowledge mining workloads |
        | Automated invoice processing; Resume parsing for recruitment; Document classification and archiving; Data extraction from legal documents | Document intelligence workloads |
        | Create new images or text based on learned patterns; AI generated music or art; Automated content generation for social media | Generative AI workloads |
- **Section 3: AI 900 - Microsoft Azure AI Fundamentals: Exploring Cognitive services**
    - **Step 01: Exploring pre trained models - Cognitive services**
        1. Cognitive services - “bring AI withing the reach of every developer”
            1. AI without building custom models
            2. Does not need machine-learning expertise
            3. Exposed as APIs
        2. Help programs see, hear, speak, search, understand (just like humans)
            1. Get intelligence from
                1. Images/videos: Customer vision, Face API, Form Recognizer
                2. Text: Text analytics, Translator text, Text-to-speech API
                3. Audio: Speecht-toText API, language understanding intelligent service - LUIS
                4. Others:
                    1. Conversations (QnA Maker, Azure bot service)
                    2. Anomaly detector service, Content moderator
    - **Cognitive services renamed to Azure AI services**
        1. Azure cognitive search has been renamed to Azure AI search
        2. Custom vision is renamed to Azure AI vision
        3. Face service is renamed to Azure AI Face detection
        4. Form recognizer is now Azure AI Document Intelligence
        5. Speech service is now Azure AI speech service
        6. Translator service is now Azure AI Translator service
    - **Step 02: Exploring vision related APIs**
        1. Vision: Get intelligence from videos & images
            1. Computer vision, Video indexer
            2. Example: [{”rectangle”: {”x”: 93, “y”: 178, “w”: 115. “h”: 237}}]. Objects, tags, background etc.
        2. Identify and analyze content within images and videos
        3. Important APIs
            1. Computer vision: Analyze content in images and videos
            2. Face API - Detect and identify people and emotions in images
            3. Custom vision: Customize image recognition to fit your business
    - **Step 03: Exploring vision - Some terminologies**
        1. Image analysis: Extract tags from image. Create text description about an image
        2. Image classification: Classify image into different groups
        3. Object detection: Identify objects in image. For each object: class of object, confidence level, coordinates of bounding box. Goes deeper than image classification
        4. Face detection: Detect human faces: Face detection and analysis: Security, tag friends on facebook, identity validation
        5. Optical character recognition (OCR): Detect text in images (license plates, invoices etc)
    - **Step 04: Getting started with computer vision API**
        
        Computer vision API: Process images and return information
        
        1. Analyze image: Extract visual features from image content
            1. Can you describe the image? (description/caption)
            2. Can you categorize the image? (tags)
            3. What is in the image? - objects/faces/celebrities/monuments with box co-ordinates
            4. What type of image is it? (clip art/line drawing)
            5. What kind of color scheme is used in the image?
            6. Does an image have mature content?
            7. Simple operations
                1. Describe image: Can you describe the image? (description/caption - multiple)
                2. Detect objects: Performs object detection on the specified image
                3. Recognize domain specific content: Identify celebrities, landmarks
                4. Tag image: Generates a list of words, or tags relevant to a image
            8. Get area of interest: most important area of the image
            9. Get thumbnail: Generates a thumbnail image (user-specified widht and height)
    - **Step 05: Demo: Creating cognitive services multi service account**
        1. Azure portal → Cognitive services → Single & multipurpose services → Create cognitive services (mutlipurpose service account) → Create new resource group → domain name (endpoint) → Review and create
    - **Step 06: Demo: Playing with custom vision API**
        1. Select your account → Keys & endpoint → Post request (header api management subscription key, resource name, request body)
    - **Step 07: Understanding computer vision API - OCR operations**
        
        OCR: Simple OCR (detects text in an image)
        
        1. Quick extraction of small amounts of text (immediate results)
        2. Results: Lines > Words (each line with bounding box co-ordinates)
        
        Read: Perform OCR on complex documents
        
        1. Optimized for text-heavy images or mutli-page documents or documents with multiple languages
        2. Executes asynchronously - Additional call to get the result
        3. Result: Pages > Lines > Words (each with bounding box coordinates)
    - **Step 08: Getting started with Face API**
        
        Face API: Advanced face detection
        
        1. Age, emotion, glasses, hair, makeup
        2. Detect human faces, find similar faces, match face with a group
        
        Improve accuracy of face identification
        
        1. Quality of images (Higer face image quality). Recommendations: frontal, clear, and face size - 200 x 200 pixels or bigger
        2. Variety of images: From diff. angles in variety of lighting setups
        
        Important concepts
        
        1. FaceList (up to 1K faces) & LargeFaceList (up to 1M faces)
        2. PersonGroup (up to 1K persons) & LargePersonGroup (up to 1M persons). Each person can have multiple face images
        
        Operations
        
        1. Detect: Detect human faces (box co-ordinates).
            1. Options to request age, gender, headPose, smile, facialHair, glasses, emotion, hair, makeup, occlusion, accessories, blur, exposure, noise and mask details
            2. Upto 100 faces in an image
        2. Find similar: Find similar faces (Find images of this specific person)
            1. Input 1: Image to match for (faceId)
            2. Input 2: Image to match against (faceId array or FaceListId or LargeFaceListId)
            3. Output: Array of the most similar faces [along with confidence]
        3. Group: Divide candidate faces (3-1000) int groups based on face similarity
            1. Input: faceIds
            2. Output: Matching group of faceIds
        4. Identify: 1-to-many identification. Find closest match of the specific query person face
            1. Input 1: Image to match for (faceId)
            2. Input 2: Image to match against (faceId array or FaceListId or LargeFaceListId)
            3. Output: Person candidates for the face (ranked by confidence)
        5. Verify: Two things you can do
            1. Do two faces belong to same person?
                1. Input: faceId1 and faceId2
            2. Identify a face belongs to a specific person?
    - **Step 09: Exploring form recogintion API**
        
        Form recognizer: Get intelligence from scanned forms
        
        1. Extract images from forms & images
        
        Operations (over pdf or image)
        
        1. Analyze business card (get analyze business card result)
        2. Analyze ID document
        3. Analyze invoice
        4. Analyze receipt
        5. Custom form: Design & extract key-value pairs, tables, and semantic values from custom documents - pdf or image
    - **Step 10: Cognitive services - Vision - Scenarios**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Recommend service: Detect and identify people and emotions in images | Face API |
        | Recommend service: Extract visual features from image content (description/tags) | Computer vision API |
        | Recommend service: Get intelligence from scanned forms  | Form recognizer API |
        | When do you use read operation to perform OCR? | Text-heavy images or multi-page documents or documents with multiple languages |
        | How can you improve accuracy of face identification? | Images - Forntal, clear, and face size - 200 x 200 pixels or bigger. Variety of images; From diff. angles in variety of lightning setups |
        
        Face API
        
        | Scenario | Solution |
        | --- | --- |
        | Recommend face API operation: Divide candidate faces (3-1000) into groups based on face similarity (Do all the faces belong to a group?) | Group |
        | Recommedn face API operation: Find closest matches of the specific query person face in a group | Identify |
        | Recommend face API operation: Do two faces belong to same person? | Verify |
        | Recommend face API operation: Does a face belong to a same person? | Verify |
    - **Step 11: Getting started - Congnitive services - Natural language processing**
        
        Get intelligence from a conversation, speech or written text in human languages
        
        1. Language: Extract meaning from unstructured text
            1. Text analytics: Detect sentiment, key phrases and named entities. Enter your message → Sentiment & key phrases, entity linking, bing entity search
            2. Translator: Translate to/from 90 languages
        2. Speech: Integrate speech into apps and services
            1. Speech service: Speech to text, Text to speech, Translation and speaker recognition
        3. Build conversations: 
            1. QnA maker: Conversational question and answer layer
            2. LUIS: Language understanding intelligent service
                1. Understand spoken (and text) commands
                2. Get info from users natural language utterances
                    1. Example: Book me a flight to cairo, Order me 2 pizzas
    - **Step 12: Exploring text analytics API**
        
        Text Analytics: Natural language processing (NLP)
        
        1. Sentinment analysis, key phrase extraction & language detection
        2. Operations
            1. Detect language: Language name, ISO 6391 code, Score (NaN-ambiguos)
            2. Entities containing personal information: returns a list of entities with personal information (”SSN”, “Bank Account” etc) in the document
            3. Key phrases: returns a list of strings denoting the key phrases. Example: Summarize a document
            4. Named entity recognition: list of general named entities in a given document. Person, location, organization, quantity, DateTime, URL, phoneNumber, IP Address etc
            5. Sentiment: detailed sentiment analysis. positive/negative review - example: 0.1 (negative), 0.9 (positive)
    - **Step 13: Exploring translator and speech API**
        
        Translator text API: text-to-text translation
        
        1. One FROM language with multiple TO languages (example: en to fr, de)
        
        API involving speech
        
        1. Speech-to-text API: Real time & batch transcription (speech recognition)
        2. Text-to-speech API: Speech synthesis
        3. Translation: Speech-to-text and speech-to-speech API
    - **Step 14: Getting started with conversational AI**
        1. Software that can carry a conversation like humans (Talk with human like a human)
        2. Use cases: Customer support, Reservations, Automation
        3. Services:
            1. QnA Maker: Convert your FAQ into a Q&A bot. You need a knowledge base (cannot talk to db)
            2. Azure bot service: Build your own bots. Enable multiple conversation channels. Channels: Alexa, Office 365 email, Facebook, Microsoft teams, Skype, Slack, Telegram
            3. Recommended architecture: QnA maker service + Azure bot service
    - **Step 15: Demo - Playing with QnA maker - 1 - Getting started**
        1. Cognitive services → QnA maker → Create
    - **Step 16: Demo - Playing with QnA maker - 1 - Setting up knowledge base**
        1. QnA maker portal → Create a knowledge base → Populate KB → Create your KB → Add QnA pair → Save and train → Test → Publish → REST API is created
    - **Step 17: Demo - Playing with Azure bot services**
        1. Create bot → SDK languages (c# and node.js) → Test in web chat → Channels
    - **Step 18: Exploring LUIS/Language understanding intelligent services**
        
        Understand spoken and text commands
        
        1. Get info from users natural language utterances. Book me a flight to cairo, Order me 2 pizzas
        2. Query → Intent and entities
        3. Detects: Intents (FoodOrder, BookFlight) and Entities (Pizzas, Cairo)
        4. Integrate with azure bot service for an end-to-end conversational solution
    - **Step 19: Cognitive services - NLP - Scenarios**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Categorize: Get intelligence from a conversation, speech or written text in human languages | Natural language processing |
        | Recommend service: Detect sentiment, key phrases and named entities from text | Text analytics API |
        | Recommend service: Detect key phrases from a document | Text analytics API |
        | Recommend service: Perform text-to-text translation | Translator text API |
        | Recommend service: Speech recognition | Speech-to-text API |
        | Recommend service: Speech synthesis | Text-to-Speech API |
        | Recommend service: Translate speech in one language to text in another | Speech-to-Text |
        | Categorize: Software that can carry a conversion like human | Converational AI |
        | Recommend service: Convert your FAQ into a question and answer REST API | QnA maker |
        | Recommend service: Build a chat bot to answer questions from your knowledge base | QnA maker service + Azure bot service |
        | QnA maker service: Can you directly connect QnA maker service to a database or an external system? | No. You need to import the question and answers first |
        | Recommend service: Understands spoken (and text) commands (get intent and entities) | LUIS: Language understanding intelligent service |
        | Recommend service: Perform sentiment analysis on reviews posted on a website | Text analytics API |
    - **Step 20: Understanding decision services / Make smarter decisions**
        
        Anomaly detector: Find anomalies
        
        1. Unusual actions, behavior or errors.
            1. Batch or real time
            2. Example usecase
                1. Find fraud
                2. Unusual transactions on a credit card
                3. Defective parts
        
        Content moderator: Detect unwanted content
        
        1. Image, text (PII, custom list) & video moderation. Return content assessment results to your systems. You can use this information to take decisions. Take content down, send to human judge
        2. Example APIs:
            1. Image-Evaluate: Returns probabilities of image having racy or adult content
            2. Text-screen: Profanity scan in 100+ languages (custom & shared blacklists)
    - **Step 20: Congnitive services - Decision services and others**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Recommend service: Unusual action, behavior or error (Unusual transactions on a credit card or fraud) | Anomaly detector |
        | Recommend service: Detect unwanted content (text, image or video) | Content moderator |
        | Recommend service: Perform profanity scan on reviews posted on a website | Content moderator |
        | Access multiple cognitive services with a single key and endpoint | Cognitive services Multi-service account |
        | What do you need to invoke a cognitive service API? | Endpoint (the HTTP address at which your service is hosted) and the key (a secret value used by client applications to authenticate themselves) |
    - **Step 21: Congnitive services - A quick review**
        1. 
    - **Quiz 2: Section Quiz**
        1. Which of these services helps you detect and identify people and emotions in images? - **Face API**
        2. Which of these Computer Vision API Operations will you use to perform OCR on Text-heavy images OR multi-page documents OR documents with multiple languages? - **Read**
        3. Which of these Face API Operations will you use to identify if two faces belong to the same person? - **Verify**
        4. Which of these Face API operations will you use to find the closest matches of a specific face in a group of faces? - **Identify**
        5. You would want to identify if a review posted on a website is positive or negative. Which operation are you performing? - **Sentiment analysis**
        6. Which of these Services will you use to translate speech in one language to text in another? - **Speech-to-Text API**
        7. You want to build a Facebook chatbot to answer questions from your knowledge base. Which services can you make use of? - **Both (QnA maker and Azure bot services)**
        8. Which of these services can be used to detect unusual actions, behavior or errors (Unusual transactions on a credit card or Fraud)? - **Anomaly detector**
- **Section 4: Generative AI in Azure**
    - **Types of AI**
        1. Strong artificial intelligence (or general AI): Intelligence of machine = Intelligence of human
            1. A machine that can solve problems, learn and plan for the future
            2. An expert at everything (including learning to play all sports and games!)
            3. Learns like a child, building on it’s own experience
        2. Narrow AI (or weak AI): Focuses on specific task
            1. Examples: Self-driving cars and virtual assistants
            2. Maching learning: Learn from data (example)
    - **Step 01 What is Generative AI?**
        1. Artificial intelligence - Create machine that simulate human-like intelligence and behavior
            1. Machine learning - Learning from examples
                1. Generative AI - Learning from example to create new content
        2. Generative AI - Generating new content
            1. Goal: Generating new content. Instead of making predictions, Generative AI focuses on creatig new data samples
            2. Examples:
                1. Text generation: Writing e-mails, essays, & poems. Generating ideas
                2. Writing code: Write,, debug & analyze programs
                3. Images generation: Creating paitings, drawings, or other forms of images
        3. How else is Generative AI different?
            1. Let’s find out!
    - **Most popular Generative AI tool today - ChapGPT**
        
        
    - **Step 02 Playing with ChatGPT - Getting Started**
        1. ChatGPT: Open AI’s Generative AI Chatbot!
        2. A Demo of ChatGPT:
    - **Step 03 Playing with ChatGPT - Coding, Learning, and Design**
        
        
    - **Step 04 Playing with ChatGPT - Exploring Technology**
        
        
    - **How is Generative AI different?**
        1. Generative AI - Generating new content
            1. Goal: Generating new content. Instead of making predictions, Generative AI focuses on creatig new data samples
            2. Examples:
                1. Text generation: Writing e-mails, essays, & poems. Generating ideas
                2. Writing code: Write,, debug & analyze programs
                3. Images generation: Creating paitings, drawings, or other forms of images
        2. How else is Generative AI different?
            1. Let’s find out!
    - **Step 05 Generative AI - Needs huge volumes of data**
        1. Generative AI models: Statistical models that learn to generate new data by analyzing existing data
            1. More data analyzed ⇒ Better new data similar to existing data
            2. Example: GPT-3 model was trained on a datasetof 500 billions words of text
        2. Datasets used include
            1. Images, text, and code scraped from open web
                1. Wikipedia
                2. Books
                3. Open source code (syntax of programming languages and the semantics of code)
                4. Conversations
    - **Step 06 Generative AI - Uses self supervised learning**
        1. Self-supervised learning: Model learns from the data itself
            1. Without requiring explicit lables or annotations
        2. How does this work?
            1. Example for text model
                1. Model tries to predict next word based on the preceeding words
                    1. Model is given example sentence: “The sun is shining and the sky is __”
                    2. Model predicts the missing word
                2. Model’s predicted word is compared to the actual word that come next
                    1. Learns from its mistakes and adjuts its internal representations
                        1. Neural networks, Loss calculation, Backpropagation etc..
                3. Repeated for all text from training dataset
        3. Model captures the relationships between words, contextual cues, and semantic meanings
            1. If prompted with “The sun is shining and the sky is, ” the model might generate
                1. “The sun is shining and the sky is clear.”
                2. “The sun is shining and the sky is bue.”
                3. “The sun is shining and the sky is filled — with fluffy clouds.”
            2. 
    - **Step 07 Key step in Generative AI for text - Predicting next word**
        1. A key step in Generative AI for text is predicting the next word
        2. During training, text based generative AI models learn the probability that a word might occur in specific context
            1. Context: “The cat sat on the”
            2. Example probabilities for next word: “mat”:0.4, “table”: 0.2, “chair”: 0.2, “moon”: 0.1
            3. Model might choose the highest probable word and go on to predict subsequent words
            4. However, you can control which of the words is chosen by controlling few parameters. temperature, top_k, top_p etc!
            
    - **Step 08 Understading deep learning**
        1. How does learning in Generative AI happen?
            1. Generative AI make use of Deep learning
        2. Let’s consider an example
            1. Young artists learn by studying styles and techniques from different art pieces
            2. With practise, they become proficient enough to create their own unique pieces
        3. Deep learning: Approach where a computer program learns from a large amount of data
            1. Starts by understanding simple patterns
            2. With time, it learns to recognize complex patterns
            3. Using the skills learned through deep learning, Generative AI can generate new content, whether it is images, music, or text
    - **Step 09 Understading loss function**
        1. Let’s consider an example:
            1. Imagin that you are teaching someone to paint
            2. You give them feedback on their work
                1. You tell them what’s good and what needs improvement
            3. The loss function in deep learning is like this feedback
                1. How far off is the current output of a model from the desired result?
                2. Goal of a deep learning system: Minimize the loss
        2. In simple terms; The loss function is a score that tells us how well the AI is performing
            1. A lower score ⇒ AI’s output is close to what we want
            2. A high score ⇒ It’s far from the target
        3. AI’s goal: Adjust and learn in a way that this score (loss) gets lower over time
    - **Step 10 Getting started with Azure Open AI**
        1. Azure Open AI: Integrate generative AI into your apps
            1. Understand and generate natural language and code
            2. Generate and edit images
            3. Convert speech into text
            4. Fine tune models
        2. Models
            1. gpt-4, gpt-3.5-turbo, dall-e, ..
        3. Demo:
            1. https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart
    - **Step 11 The AI turnmoil - My view point**
        
        
    
- **Section 5: AI 900 - Microsoft Azure AI Fundamentals: Custom vision & Azure machine learning**
    - **Course downloads - Custom vision - Images**
        
        
    - **Step 01 - Building ML Models - Custom Vision 1**
        1. Custom vision: Create custom models using your own images
        2. Project types
            1. Classification: Predict labels for an image. Two classification types. 1. Multilabel (Multiple tags per image). 2. Multiclass (Single tag per image).
            2. Object detection: Return coordinates of objects in an image
        3. Best practices
            1. Pick the domain closest to your scenario: Different domains are available for classification and object detection projects
            2. Sufficient images (Add more images to improve accuracy)
            3. From different angles
        4. Demo - Image classification
            1. Add images and tag them
            2. Traing the model
    - **Step 02 - Building ML Models - Custom Vision 2**
        1. Training completed - Quick test
    - **Step 03 - Creating ML Models - Features and labels**
        
        Understanding machine learning
        
        1. Traditional programming: Based on rules
        2. Machine learning: Learning from examples
        
        Creating machine learning models - Features and labels
        
        1. Goal of machine learning: Create good model
            1. Give inputs to a model
            2. Model returns the prediction
            3. Inputs are called features
            4. Predicition is called label
        2. Example: House price prediction model
            1. Label: Price
            2. Features
                1. area: Total area of house (m^2)
                2. rooms: No. of rooms
                3. bedrooms: No. of bedrooms
                4. furniture: Is it furnished?
                5. floor: which floor?
                6. age: How many years?
                7. balcony: has balcony or not?
        3. Used car price prediction model
            1. Label: price
            2. Features: manufacturer, year, model, age, condition, cylinders, location
        4. Spam email classification model
            1. Label: isSpam
            2. Features: sender, subject, content
        5. Grant a loan model
            1. Label: shouldWeGrantLoan
            2. Features: doesOwnCar, doesOwnRealEstate, creditScore, isMarried, doesHaveChildren, totalIncome, totalCredit
    - **Step 04 - Creating ML Models - Choosing techniques**
        
        Supervised learning: Features and label
        
        1. Label is a numeric value with a range of possibilities ⇒ Regression
            1. Example: Used car price prediction, House price prediction, Predicting sea level, Predicting no of vechicles that use a specific high way
            2. How much will it rain tomorrow?
        2. Label has limited set of possibilities (YES or NO, 0 or 1, Type 1 or Type 2 or Type 3) ⇒ Classification
            1. Spam email, Grant a loan, Determine the type of cloud
            2. Will it rain today?
        3. Summary: Supervised machine learning models
            1. Classification: Predicting category
            2. Regression: Predicting numeric value
        
        Unsupervised learning: Labels
        
        1. Clustering: Divide customers into groups
            1. Group similar entities based on their features
    - **Step 05 - Machine learning fundamentals - Scenarios**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Categorize into features and labels for house price prediction: price, area, rooms, age | price is label. Others can be features |
        | Categorize into features and labels for used vehicle price prediction: manufacturer, year, model, age, condition, cylinders, location, price | price is label. Others can be features |
        | Categorize: Used car price prediction | Regression |
        | Categorize: Spam email indentification | Classification |
        | Categorize: Predict amount of rainfall in the next year | Regression |
        | Categorize: Should we grant a loan? | Classification |
        | Categorize: Identify the type of vehicle in an image | Classification |
        | Categorize: Find a specific dance form in a video | Classification |
        | Categorize: Divide customers into groups | Clustering |
    - **Step 06 - Creating machine learning models - Steps**
        1. Obtain data
        2. Clean data
        3. Feature engineering: Identify features and labels
        4. Create a model using the dataset and the ML algorithm
        5. Evaluate the accuracy of the model
        6. Deploy the model for use
    - **Step 07 - Understanding machine learning terminology**
        
        Process
        
        1. Training: The process of creating a model
        2. Evaluation: Is the model working?
        3. Inference: Data used to create, validate & test the model
        
        Dataset: Data used to create, validate, and test the model
        
        1. Features: Inputs
        2. Label: Output/Prediction
        3. Dataset types
            1. Training Dataset: Dataset used to create a model
            2. Validation Dataset: Dataset used to validate the model (and choose the right algorithm) - Model evaluation
            3. Testing Dataset: Dataset used to do final testing before deployment
    - **Step 08 - ML stages and terminology - Scenarios**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Determine stage: You remove data having null values from your dataset | Clean data (Data preparation) |
        | Determine stage: Normalize or split data ino multiple featurs | Feature engineering |
        | Determine stage: You evaluate the accuracy metrics of a model | Model evaluation |
        | Terminilogy: Using model to do predictions in production | Inference |
        | Terminology: The process of creating a model | Training |
        | Terminology: Dataset used to train or create a model | Training dataset |
        | Terminology: Dataset used to evaluate a model | Validation dataset |
        | Terminology: Dataset used to do final testing before deployment | Testing dataset |
    - **Step 09 - Getting started with Azure Machine Learning**
        1. Azure machine learning: Simplifies creation of your models
            1. Maage data, code, compute, models etc
            2. Prepare data
            3. Train models
            4. Publish models
            5. Monitor models
        2. Multiple options to create models
            1. Automated machine learning: Build custom models with minimum ML experience
            2. Azure machine learning designer: Enables no-code development of models
                1. Build your own models: Data scientist
        3. Demo
            1. Machine learning → Workspace
    - **Step 10 - Demo - Automated machine learning - Dataset and compute**
        
        Demo
        
        1. Machine learning workspace → Azure machine learning studio
        2. Register data → Traing models → Evaluate models → Deploy models
        3. Model Creation
            1. Notebooks
            2. Automated ML
            3. Designer
        4. Compute environment to run the workloads. Compute → Compute clusters → Create compute
        5. Datasets → Create dataset
        6. Datasets → Open dataset → Profile
    - **Step 11 - Demo - Setting up Automated ML training**
        1. Automated ML → Start now → New Automated ML run (Finds the best model based on your data without writing a single line of code)
        2. Select dataset → Configure run (Experiment, Target column, and select compute cluster)→ Select task and settings (Classification, regression, and time series forescasting) → → Finish
        3. View additional configuration (select metric to tune the model) 
            1. Blocked algorithms because automated ML creates multiple model for each algorithm and find the best one
            2. Exit criterion - Time
        4. View featurization settings
            1. Feature type
            2. Impute with
    - **Step 12 - Demo - Exploring automated ML training**
        1. Automated ML → Experiment → Models → Normalized root mean square error
        2. Model → Select algorithm → Deploy model (Compute type and compute name)
        3. Endpoint → Model in action → Deployment state as Healthy → Test → Provide inputs and gets predicitons
        4. REST API endpoint is also available
    - **Step 13 - Demo - Creating azure machine learning pipeline**
        1. Designer → Start now → Create a new pipeline → Canvas
        2. Modules → Datasets, Data input and output, Data transformations (split data)
        3. Split data - Split data into two. Training and evaluation
        4. Algorithms - Regression, Classification, and clustering
        5. Linear regression
        6. Train model
        7. Score model
        8. Evaluate model
    - **Step 14 - Demo - Model Evaluation - Regression models**
        1. Experiements → Run → Evaluation results → Publish the pipline and create new endpoint
        2. Mean absolute error (MAE): How close is a prediction to actual value? Lower the better
        3. Mean squared error (MSE): Average of squares of the distance between actual and predicted. When you want to penalize large prediction errors (housing value predictions). Lower the better
            1. Alternative: Root mean squared error: Sqaure root of root of MSE. Lower the better
    - **Step 15 - Model Evaluation - Classification models**
        
        Terminology
        
        1. Predicted label: What’s predicted?
        2. True label: what’s expected?
        3. Confusion matrix: Matrix matching predicted label vs true label
        
        Different usecases have different needs
        
        1. Examples: spam, fraud, sick patient detection
        
        Metrics
        
        1. Accuracy: Proportion of accurate results to total cases
        2. Precision: (True positive) / (True positive + False positive)
        3. Recall: (True positive) / (True positive + False negative)
        4. F1 score: 2 ((Precision recall) / (Precision + recall)). When you need balance between precision and recall
    - **Step 16 - ML Model Evaluation - Scenarios**
        
        
        | Scenarion | Solution |
        | --- | --- |
        | Model evaluation terminology: What’s predicated? | Predicted label |
        | Model evaluation terminology: What’s expected? | True label |
        | Model evaluation terminology: Matrix matching predicted label vs true label | Confusion matrix |
        | Model evaluation metrics for classification | Accuracy, Precision, Recall, F1 Score |
        | Model evaluation metrics for regression | Mean absolute error (MAE), Mean squared error (MSE), Root mean squared error (RMSE) |
    - **Step 17 - Azure machine learning - Terminology**
        
        Studio: Website for azure machine learning
        
        Workspace: Top-level resource for azure machine learning
        
        1. Azure machine learning designer: drag and drop interface to create your ML workflows (canvas)
            1. Pipelines: Reusable workflows (training and re-training)
            2. Datasets: Manage your data
            3. Module: An algorithm to run on your data
                1. Data preparation: Data transformation, Feature selection
                2. Machine learning algorithms: Regression, Classification, and Clustering
                3. Building and evaluating models: Model training, model scoring and evaluation
            4. Compute:
                1. Compute instances: Development machines (CPU or GPU instances) for data engineers, and data scientists
                    1. Pre-configured with tools such as Jupyter, ML packages etc
                2. Compute clusters: Training machines
                    1. Single or multi-node compute cluster for your training
                3. Inference clusters: Deployment machines
                    1. Deploy your model to Azure kubernetes service or azure container instances
                4. Attached compute: Use HDInsight cluster, a virtual machine, or a databricks cluster as target for azure machine learning workspace
    - **Step 19 - Building custom ML models in Azure - Scenarios**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Recommended service: Create custom models using your own images | Custom vision |
        | Terminology: Website for azure machine learning | Azure machine learning studio |
        | Drag-and-drop interface to create your ML workflows | Azure machine learning designer |
        | Reusable workflows (training and re-training) | Pipelines |
        | Data used for training | Dataset |
        | What are the components that can be dragged on to canvas to build a pipeline? | Modules |
        | What are training machines for azure machine learning called? | Compute clusters |
        | What are deployment machines for azure machine learning called? | Inference clusters |
        | Why do you split data when you build a ML model? | To use a part of training and rest of the data for validation for model |
        | How can you consume an Azure machine learning model? | Publish it and access it as a web service (REST API endpoint) |
        | Languages popularly used with ML | Python and R |
        | Store and version your models. Organize and keep track of your trained models | Model registration |
    - **Quiz 3: Section quiz**
        1. Which of these project types in Custom Vision allow you to select Domain? - **Classification and object detection**
        2. Which of these project types in Custom Vision has two classification types - Multilabel and Multiclass? - **Classification**
        3. Identify label for house price prediction model: price, area, rooms, age - **Price**
        4. Categorize ML Problem: Used Car Price Prediction - **Regression**
        5. Categorize ML Problem: Analyze a Traffic Light image to find the signal - Red or Green or Amber - **Classification**
        6. Which dataset is used to create a model? - Training dataset
        7. **Accuracy** metric is used to evaluate models created for: - **Classification**
        8. Root Mean Squared Error(RMSE) metric is used to evaluate models created for: - **Regression**
- **Section 6: AI 900 - Microsoft Azure AI Fundamentals: Responsible AI principles**
    - **Step 01 - Most important AI considerations - Responsible AI Principles**
        
        Challenges in building AI solutions
        
        1. Importance of datasets
            1. What if the data has a bias? (Bias can affect results). Solutions may not work for everyone
            2. Obtaining data
        2. Evolving field
            1. What if an AI system causes error? Accident made by a self driving car. Errors may cause harm
            2. Scarcity of skills (Data Scientist, …)
        3. ML lifecycle (MLOps)
        4. Security (What if the data used to build the model is exposed?)
        5. Explainability of model (Users must trust a complex system)
        6. Who will face the consequences? Who’s liable for AI-driven decisions?
        
        Responsible AI principles
        
        AI without unintended negative consequences
        
        1. Fairness - Fair to all groups of people. System’s decisions don’t discriminate or run a gender, race, sexual orientation, or religion bias towards a group or individual. Data should reflect diversity, model should evolve with time
        2. Reliability and safety - Continues working under high loads, unexpected situations etc. What happens in bad whether? What if GPS is down? What happens if data is bad? Test, test, and test
        3. Privacy and security - Of people and data! (information and controls). Important consideration from day zero
        4. Inclusiveness - Nobody left out. Violation: Leaving out a certain group of people (ex: people with disabilities)
        5. Transparency - Explainability, debuggability. Clear explanation to users
        6. Accountability - Meets ethical and legal standards. AI is not the final decision maker, An enterprise, a team or a person is. 
    - **Step 02 - AI considerations - Scenarios**
        
        
        | Scenario | Solution |
        | --- | --- |
        | Identify violated principle: You find that a ML model does not grant loans to people of certain gender | Fairness |
        | Identify violated principle: More accidents caused by a self driving car in bad whether | Reliability and safety |
        | Identify related principle: Securing data used to create a model | Privacy and security |
        | Identify related principle: Making sure that the dataset used does not have any errors (missing values etc) | Reliability and safety |
        | Identify violated principle: People with disabilities cannot use a specifi AI solution | Inclusiveness |
        | Identify related principle: Giving your customers control/choice over the data that is used by your AI system | Privacy and security |
        | Identify related principle: Ensuring that an AI system works reliably under unexpected situaion | Reliability and safety |
        | Identify violated principle: You do not know how a AI system reached a specific inference | Transparency |
        | Identify related principle: Ensuring that there is sufficient information to debug problems with an AI system | Transparency |
        | Identify related principle: Having a team that can override decision made by an AI system | Accountability |
    - **Quiz 4: Section quiz**
        1. Identify violated principle: You find that an ML model grants loans disproportionately to people of a certain gender - **Fairness**
        2. Identify related principle: Ensuring that there is sufficient information to debug problems with an AI system - **Transparency**
        3. Identify related principle: Giving your customers control/choice over the data that is used by your AI system - **Privacy and security**
    
- **Section 7: AI 900 - Microsoft Azure AI Fundamentals: You are all set**
    - **01 AI 900 - Azure AI Fundamentals - Get ready**
        
        Certification exam
        
        1. Exam AI 900 - Microsoft azure AI fundamentals [https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-fundamentals/?practice-assessment-type=certification](https://learn.microsoft.com/en-us/credentials/certifications/azure-ai-fundamentals/?practice-assessment-type=certification)
        2. Different types of multiple choice questions
            1. Type 1 - Single answer
            2. Type 2 - Multiple answers
        3. No penalty for wrong answers
        4. 40 questions and 60 minutes
    - **02 AI 900 - Azure AI Fundamentals - Congratulations**
        
        
    
- **Section 8: Practise test - Azure Certification AI 900 - Azure AI Fundamentals**
    - **Practise Test 1: Practise test - Azure Certification AI 900 - Azure AI Fundamentals**
        1. Which of these Azure services helps you to use AI without needing to build custom models? **- Azure cognitive services**
        2. Which of these Azure services helps you build a custom image classification model to fit your business needs (example: for identifying machine parts)? **- Custom vision**
        3. Which of these Face API operations helps you detect if the person in an image is wearing glasses? **- Detect**
        4. Which of these Face API operations helps you to find if two faces belong to the same person? **- Verify**
        5. You would want identify if a review posted on a website is positive or negative. Which operation are you performing? **- Sentiment Analysis**
        6. Which of these Services will you use to translate speech in one language to text in another? **- Speech (Speech-to-Text API)**
        7. You want to build a website chat bot to answer questions from your knowledge base. Which services can you make use of? **- Both (QnA maker service and Azure bot service)**
        8. Which of these services can be used to detect unusual transactions on a credit card? **- Anomaly detector**
        9. Which of these services can be used to detect profanity in reviews posted on a shopping website? **- Content moderator**
        10. Which of these do you need to make a call to an Azure Cognitive Service? **- Both (API endpoint and key)**
        11. Categorize: You want to create an ML model to predict if it will rain tomorrow? **- Classification**
        12. Categorize: You want to create an ML model to predict how much it will rain tomorrow? **- Regression**
        13. During which ML stage would you perform the following operation: Normalize or split data into multiple features **- Feature engineering**
        14. Which of these datasets is used to evaluate a model? **- Validation dataset**
        15. Which of these metrics can be used to evaluate a model for classification? **- Accuracy, Precision, Recall**
        16. Which of these metrics can be used to evaluate a model for regression? **- Mean absolute error (MAE) or root mean squared error**
        17. Confusion Matrix is used for evaluation of: **- Classification models**
        18. During evaluation of a two-way classification model, there are 25 false positives. What does that mean? - There are 25 values which should have been negative that are identified as positive by model
        19. Which of these programming languages are mostly used in Machine Learning? **- Python, R**
        20. Which of these are used as compute for training in Azure Machine Learning? - **Compute clusters**
        21. Identify violated principle: You find that a ML model does not grant loans to people of certain gender - **Fairness**
        22. Identify violated principle: More accidents caused by a self driving car in bad weather **- Reliability and safety**
        23. Identify related principle: Making sure that the dataset used does not have any errors (missing values etc) **- Reliability and safety**
        24. Speech synthesis is: **- Converting text to speech**
        25. Which of these modules do you use in Azure Machine Learning to split data into training set and validation set? - **Split data**
        26. True or False: Computer Vision can be used to create a custom image classification model **- False**
        27. Which of these channels are supported by Azure Bot Service? **- All the above**
        28. You have used Text Analytics to perform Natural language processing (NLP) on a news article. You have identified the people, locations, organizations and date time values present in the article. Which of these have you performed? **- Entity recognition**
        29. You want to identify vehicles in an image. Which of these are you doing? - **Object detection**
        30. Which of these services can be used to convert user commands (text and voice) into intent and entities? - **LUIS (Language understanding intelligent service)**
        31. Which of these project types in Custom Vision allow you to select Domain? **- Both (Classification and Object detection)**
        32. Which of these project types in Custom Vision has two classification types - Multilabel and Multiclass? - **Classification**
        33. Which of these can be a feature when creating a House Price Prediction Model? **- Area of the house, age of the house**
        34. Which of these is the label in a Spam Email Classification Model? **- isSpam**
        35. Identify related principle: Having a team that can override decision made by an AI system - **Accountability**
        36. Which of these services can be used to detect people and emotions in images? - **Face API**
        37. Which of these services help you scan a receipt and identify the key value pairs and tables? **- Form recognizer API**
        38. Which of these do NOT follow under the umbrella of Natural Language Processing? - **OCR**
        39. True or False: Automated machine learning helps you build custom models with minimum ML expertise - **True**
        40. True or False: Accuracy is the only metric used to evaluate all classification models **- Fasle**
        41. True or False: You can consume an Azure Machine Learning model by publishing it and accessing it as a web service (REST API endpoint) **- True**