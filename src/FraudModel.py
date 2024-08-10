import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import BertTokenizer, BertModel



class FraudDetectionDataset(Dataset):
    def __init__(self, html_features, labels):
        self.html_features = html_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.html_features[idx], self.labels[idx]

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FraudDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
# Assuming html_features is a tensor of preprocessed HTML content and labels is a tensor of fraud labels
# html_features.shape = (num_samples, input_dim), labels.shape = (num_samples,)

def tokenize_html(html_content):
    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Example: Tokenizing text content extracted from HTML
    tokens = tokenizer.tokenize(html_content)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the text and get BERT embeddings
    inputs = tokenizer(html_content, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings for the [CLS] token, representing the entire sentence
    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # convert to tensor
    sentence_embedding = torch.tensor(sentence_embedding, dtype=torch.float32)

    return sentence_embedding

def generate_dummy_data():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    num_samples = 10000  # Number of samples
    input_dim = 768     # Dimension of each feature vector

    # Generate random HTML features
    # Assuming the HTML content has been transformed into numerical features
    # Here, we'll just use random numbers to simulate this
    html_features = np.random.rand(num_samples, input_dim)

    # Generate random binary labels (0 or 1) for fraud detection
    labels = np.random.randint(0, 2, size=(num_samples,))

    # Convert to PyTorch tensors
    html_features_tensor = torch.tensor(html_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    return html_features_tensor, labels_tensor

def train():
    num_epochs = 10000
    for epoch in range(num_epochs):
        for i, (html_data, label) in enumerate(dataloader):
            # Forward pass
            outputs = model(html_data)
            loss = criterion(outputs, label.unsqueeze(1).float())
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate():
    # Assuming test_html_features and test_labels are tensors for the test set
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        test_outputs = model(test_html_features)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == test_labels.unsqueeze(1)).sum().item() / test_labels.size(0)
        
        print(f'Accuracy on test set: {accuracy:.4f}')


if __name__ == "__main__":
    html_features, labels = generate_dummy_data()
    test_html_features, test_labels = generate_dummy_data()
    input_dim = html_features.shape[1]
    hidden_dim = 64  # You can adjust the hidden layer size

    dataset = FraudDetectionDataset(html_features, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = FraudDetectionModel(input_dim, hidden_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # train()
    # evaluate()
    max_len = 512
    content ="""<form id=\"checkoutBillingForm\" novalidate=\"\" style=\"flex: 1 1 0%;\"><div class=\"css-1rc289k\"><div class=\"css-188a0vm\"><div class=\"css-18rqyaq\"><h1 class=\"css-1p4a4l4\">Customer &amp; Shipping Information</h1><a class=\"css-u5yvmo\" aria-label=\"Edit your shipping address\" hreflang=\"en\" href=\"/checkout/shipping\">Edit</a></div><div class=\"css-vurnku\"><p class=\"css-1ozwjwc\">ljakfsjd@gmail.com</p><p class=\"css-11kedix\">segklawefm;lw'; ejnglkfme</p><p class=\"css-11kedix\"></p><p class=\"css-11kedix\">25 Sir Williams Lane</p><p class=\"css-11kedix\"></p><p class=\"css-11kedix\">Etobicoke, ON M9A 1T9 Canada</p><p class=\"css-11kedix\">16473034243</p></div></div><div class=\"css-vurnku\"><fieldset data-testid=\"billing-payment\" class=\"css-14pgucx\"><legend class=\"css-1jkqrpx\">Billing and payment</legend><div class=\"css-gnqbje\"><div class=\"css-0\" data-reach-accordion=\"\"><div data-testid=\"AccordionRadioItem\" class=\"css-11d2m22\" data-reach-accordion-item=\"\" data-state=\"open\"><button aria-controls=\"panel--:r13:--0\" aria-expanded=\"true\" role=\"radio\" aria-checked=\"true\" class=\"css-1myo5ch\" data-reach-accordion-button=\"\" data-state=\"open\" id=\"button--:r13:--0\"><svg xmlns=\"http://www.w3.org/2000/svg\" fill=\"currentcolor\" viewBox=\"0 0 24 24\" class=\"css-1lehszp\"><path d=\"M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5m0-5C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2m0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8\"></path></svg><div class=\"css-1rxckop\"><span class=\"css-74kml7\">Credit Card</span><div class=\"css-123llyo\"><svg xmlns=\"http://www.w3.org/2000/svg\" fill=\"none\" viewBox=\"0 0 58 37\" aria-label=\"Visa\" class=\"css-1g2iw96\"><path fill=\"#1A1F70\" d=\"M55.093 37H2.907A2.895 2.895 0 0 1 0 34.093V3.178A2.895 2.895 0 0 1 2.907.271h52.186A2.895 2.895 0 0 1 58 3.178v30.915C58 35.708 56.662 37 55.093 37\"></path><path fill=\"#fff\" d=\"m28.84 13.053-2.4 11.166h-2.907l2.4-11.167zm12.135 7.244 1.522-4.2.877 4.2zm3.23 3.922h2.676l-2.353-11.167h-2.446c-.554 0-1.015.324-1.246.831L36.5 24.219h3.045l.6-1.661h3.692l.369 1.66zm-7.521-3.645c0-2.954-4.06-3.092-4.06-4.43 0-.415.368-.83 1.245-.923.415-.046 1.569-.092 2.86.508l.508-2.353c-.692-.231-1.569-.508-2.722-.508-2.86 0-4.845 1.523-4.891 3.691 0 1.615 1.43 2.492 2.538 3.046s1.522.876 1.476 1.384c0 .738-.877 1.061-1.707 1.107-1.477 0-2.307-.415-2.953-.692l-.554 2.4c.692.323 1.938.6 3.23.6 3 0 4.983-1.477 5.03-3.83m-11.997-7.522-4.66 11.167H16.98l-2.307-8.906c-.138-.553-.277-.738-.692-.969-.692-.369-1.846-.738-2.86-.968l.091-.323h4.938c.646 0 1.2.415 1.338 1.153l1.2 6.46 2.998-7.614h3z\"></path></svg><svg xmlns=\"http://www.w3.org/2000/svg\" fill=\"none\" viewBox=\"0 0 58 37\" aria-label=\"MasterCard\" class=\"css-1g2iw96\"><path fill=\"#0F1822\" d=\"M55 0H3a3 3 0 0 0-3 3v31a3 3 0 0 0 3 3h52a3 3 0 0 0 3-3V3a3 3 0 0 0-3-3\"></path><path fill=\"#F26522\" d=\"M35.774 6.685H22.682v23.529h13.092z\"></path><path fill=\"#E52423\" d=\"M23.546 18.45c0-4.774 2.235-9.025 5.715-11.765a14.9 14.9 0 0 0-9.247-3.198C11.75 3.487 5.05 10.186 5.05 18.45s6.699 14.962 14.963 14.962c3.49 0 6.702-1.196 9.247-3.198-3.48-2.743-5.715-6.991-5.715-11.764z\"></path><path fill=\"#F99F1C\" d=\"M52.043 27.718v-.483h.195v-.098h-.494v.098h.195v.483zm.96 0v-.58h-.153l-.174.4-.174-.4h-.153v.58h.108v-.438l.163.379h.111l.164-.379v.438zM38.51 3.487a14.9 14.9 0 0 0-9.248 3.198 14.94 14.94 0 0 1 5.715 11.765c0 4.773-2.235 9.021-5.715 11.76a14.9 14.9 0 0 0 9.247 3.199c8.264 0 14.963-6.7 14.963-14.963-.004-8.26-6.7-14.959-14.963-14.959z\"></path></svg><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"38\" height=\"24\" viewBox=\"0 0 38 24\" aria-label=\"American Express\" class=\"css-1g2iw96\"><g fill=\"none\"><path fill=\"#000\" d=\"M35 0H3C1.3 0 0 1.3 0 3v18c0 1.7 1.4 3 3 3h32c1.7 0 3-1.3 3-3V3c0-1.7-1.4-3-3-3\" opacity=\"0.07\"></path><path fill=\"#006FCF\" d=\"M35 1c1.1 0 2 .9 2 2v18c0 1.1-.9 2-2 2H3c-1.1 0-2-.9-2-2V3c0-1.1.9-2 2-2z\"></path><path fill=\"#FFF\" d=\"m8.971 10.268.774 1.876H8.203zm16.075.078h-2.977v.827h2.929v1.239h-2.923v.922h2.977v.739l2.077-2.245-2.077-2.34zm-14.063-2.34h3.995l.887 1.935L16.687 8h10.37l1.078 1.19L29.25 8h4.763l-3.519 3.852 3.483 3.828h-4.834l-1.078-1.19-1.125 1.19H10.03l-.494-1.19h-1.13l-.495 1.19H4L7.286 8h3.43zm8.663 1.078h-2.239l-1.5 3.536-1.625-3.536H12.06v4.81L10 9.084H8.007l-2.382 5.512H7.18l.494-1.19h2.596l.494 1.19h2.72v-3.935l1.751 3.941h1.19l1.74-3.929v3.93h1.458zm9.34 2.768 2.531-2.768h-1.822l-1.601 1.726-1.548-1.726h-5.894v5.518h5.81l1.614-1.738 1.548 1.738h1.875l-2.512-2.75z\"></path></g></svg><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"34\" height=\"30\" fill=\"none\" viewBox=\"0 0 47 30\" aria-label=\"Discover\" class=\"css-11oxnf2\"><path fill=\"#F58220\" d=\"M21.4 12c0-1.9 1.6-3.4 3.5-3.4 2 0 3.5 1.5 3.5 3.4s-1.6 3.4-3.5 3.4c-2 0-3.5-1.5-3.5-3.4M10.2 29.5c23.6-4 36.4-13.1 36.4-13.1v13.1z\"></path><path stroke=\"#000\" stroke-width=\"0.5\" d=\"M.25.25h46.5v29.5H.25z\"></path><path fill=\"#000\" d=\"M13.4 11.2c-.8-.3-1-.4-1-.8s.4-.7 1-.7c.4 0 .7.2 1.1.5l.7-.9c-.5-.4-1.2-.8-1.9-.8-1.1 0-2 .8-2 1.9 0 .9.4 1.3 1.7 1.8l.3.118c.322.126.522.204.6.282.2.2.4.4.4.7 0 .5-.4.9-1 .9s-1.1-.3-1.4-.9l-1 .7c.6.9 1.3 1.3 2.2 1.3 1.3 0 2.2-.9 2.2-2.2 0-.9-.5-1.4-1.9-1.9\"></path><path fill=\"#000\" fill-rule=\"evenodd\" d=\"M3 8.7h1.9c2 0 3.5 1.3 3.5 3.1 0 1-.4 2-1.2 2.6-.6.6-1.3.8-2.3.8H3zm1.7 5.4c.8 0 1.3-.1 1.7-.5s.7-1 .6-1.6c0-.7-.3-1.3-.7-1.7-.4-.3-.9-.5-1.7-.5h-.3v4.3z\" clip-rule=\"evenodd\"></path><path fill=\"#000\" d=\"M10.3 8.7H9v6.5h1.3zM19.1 15.3c-1.9 0-3.4-1.5-3.4-3.4 0-1.8 1.5-3.3 3.4-3.3.5 0 1.1.1 1.6.4v1.5c-.5-.5-1-.7-1.6-.7-1.2 0-2.2.9-2.2 2.2s.9 2.2 2.2 2.2c.6 0 1.1-.3 1.6-.8v1.5c-.5.3-1.1.4-1.6.4M31.1 13.1l-1.7-4.3H28l2.8 6.7h.7l2.8-6.7H33zM38.5 15.2h-3.6V8.7h3.6v1.1h-2.3v1.5h2.3v1.1h-2.3v1.7h2.3z\"></path><path fill=\"#000\" fill-rule=\"evenodd\" d=\"M43.5 10.6c0-1.2-.8-1.9-2.3-1.9h-1.9v6.5h1.3v-2.6h.2l1.8 2.6h1.6l-2.1-2.8c.9-.1 1.5-.8 1.4-1.8M41 11.7h-.4v-2h.4c.8 0 1.2.3 1.2 1 0 .6-.4 1-1.2 1M44.146 9.554h-.11v-.503h.177c.117 0 .193.048.193.151v.003c0 .076-.043.117-.103.135l.133.214h-.115L44.2 9.36h-.053zm0-.418v.147h.065q.086 0 .087-.074v-.002c0-.053-.032-.071-.087-.071z\" clip-rule=\"evenodd\"></path><path fill=\"#000\" fill-rule=\"evenodd\" d=\"M44.721 9.308a.503.503 0 0 1-.51.509.503.503 0 0 1-.51-.506c0-.283.225-.511.51-.511.29 0 .51.225.51.508m-.929.003c0 .253.184.423.419.423a.41.41 0 0 0 .418-.426.405.405 0 0 0-.418-.425c-.244 0-.419.177-.419.428\" clip-rule=\"evenodd\"></path></svg></div></div></button><div role=\"region\" aria-labelledby=\"button--:r13:--0\" class=\"css-k008qs\" data-reach-accordion-panel=\"\" data-state=\"open\" id=\"panel--:r13:--0\"><div class=\"css-b0oq40\" style=\"height: auto; opacity: 1;\"><div class=\"css-1q4z5sm\"><div class=\"css-vurnku\"><div class=\"css-84yc0d\"><div id=\"cardNumber\" placeholder=\"\" class=\"css-1iucqta isFocused isBlurred braintree-hosted-fields-valid\"><iframe src=\"https://assets.braintreegateway.com/web/3.97.1/html/hosted-fields-frame.min.html#0b510647-3498-46bc-a0a7-31b03000fee0\" frameborder=\"0\" allowtransparency=\"true\" scrolling=\"no\" type=\"number\" name=\"braintree-hosted-field-number\" title=\"Secure Credit Card Frame - Credit Card Number\" id=\"braintree-hosted-field-number\" style=\"border: none; width: 100%; height: 100%; float: left;\"></iframe><div style=\"clear: both;\"></div></div><label class=\"paymentInputLabel css-pb5mm5\">Card Number</label></div><div class=\"css-84yc0d\"><div id=\"cvv\" placeholder=\"\" class=\"css-1iucqta isFocused isBlurred braintree-hosted-fields-valid\"><iframe src=\"https://assets.braintreegateway.com/web/3.97.1/html/hosted-fields-frame.min.html#0b510647-3498-46bc-a0a7-31b03000fee0\" frameborder=\"0\" allowtransparency=\"true\" scrolling=\"no\" type=\"cvv\" name=\"braintree-hosted-field-cvv\" title=\"Secure Credit Card Frame - CVV\" id=\"braintree-hosted-field-cvv\" style=\"border: none; width: 100%; height: 100%; float: left;\"></iframe><div style=\"clear: both;\"></div></div><label class=\"paymentInputLabel css-pb5mm5\">CVV</label></div><div class=\"css-84yc0d\"><div id=\"exp\" placeholder=\"\" class=\"css-1iucqta isFocused braintree-hosted-fields-valid isBlurred\"><iframe src=\"https://assets.braintreegateway.com/web/3.97.1/html/hosted-fields-frame.min.html#0b510647-3498-46bc-a0a7-31b03000fee0\" frameborder=\"0\" allowtransparency=\"true\" scrolling=\"no\" type=\"expirationDate\" name=\"braintree-hosted-field-expirationDate\" title=\"Secure Credit Card Frame - Expiration Date\" id=\"braintree-hosted-field-expirationDate\" style=\"border: none; width: 100%; height: 100%; float: left;\"></iframe><div style=\"clear: both;\"></div></div><label class=\"paymentInputLabel css-pb5mm5\">Expiration Date</label></div></div></div></div></div></div><div data-testid=\"AccordionRadioItem\" class=\"css-1u09ivb\" data-reach-accordion-item=\"\" data-state=\"collapsed\"><button aria-controls=\"panel--:r13:--1\" aria-expanded=\"false\" role=\"radio\" aria-checked=\"false\" aria-label=\"PayPal\" class=\"css-1myo5ch\" data-reach-accordion-button=\"\" data-state=\"collapsed\" id=\"button--:r13:--1\"><svg xmlns=\"http://www.w3.org/2000/svg\" fill=\"currentcolor\" viewBox=\"0 0 24 24\" class=\"css-1s9i1vx\"><path d=\"M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2m0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8\"></path></svg><div class=\"css-1oobfo3\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"124\" height=\"33\" viewBox=\"0 0 124 33\" class=\"css-v7v99c\"><path fill=\"#253B80\" d=\"M46.211 6.749h-6.839a.95.95 0 0 0-.939.802l-2.766 17.537a.57.57 0 0 0 .564.658h3.265a.95.95 0 0 0 .939-.803l.746-4.73a.95.95 0 0 1 .938-.803h2.165c4.505 0 7.105-2.18 7.784-6.5.306-1.89.013-3.375-.872-4.415-.972-1.142-2.696-1.746-4.985-1.746M47 13.154c-.374 2.454-2.249 2.454-4.062 2.454h-1.032l.724-4.583a.57.57 0 0 1 .563-.481h.473c1.235 0 2.4 0 3.002.704.359.42.469 1.044.332 1.906m19.654-.079h-3.275a.57.57 0 0 0-.563.481l-.145.916-.229-.332c-.709-1.029-2.29-1.373-3.868-1.373-3.619 0-6.71 2.741-7.312 6.586-.313 1.918.132 3.752 1.22 5.031.998 1.176 2.426 1.666 4.125 1.666 2.916 0 4.533-1.875 4.533-1.875l-.146.91a.57.57 0 0 0 .562.66h2.95a.95.95 0 0 0 .939-.803l1.77-11.209a.568.568 0 0 0-.561-.658m-4.565 6.374c-.316 1.871-1.801 3.127-3.695 3.127-.951 0-1.711-.305-2.199-.883-.484-.574-.668-1.391-.514-2.301.295-1.855 1.805-3.152 3.67-3.152.93 0 1.686.309 2.184.892.499.589.697 1.411.554 2.317m22.007-6.374h-3.291a.95.95 0 0 0-.787.417l-4.539 6.686-1.924-6.425a.95.95 0 0 0-.912-.678h-3.234a.57.57 0 0 0-.541.754l3.625 10.638-3.408 4.811a.57.57 0 0 0 .465.9h3.287a.95.95 0 0 0 .781-.408l10.946-15.8a.57.57 0 0 0-.468-.895\"></path><path fill=\"#179BD7\" d=\"M94.992 6.749h-6.84a.95.95 0 0 0-.938.802l-2.766 17.537a.57.57 0 0 0 .562.658h3.51a.665.665 0 0 0 .656-.562l.785-4.971a.95.95 0 0 1 .938-.803h2.164c4.506 0 7.105-2.18 7.785-6.5.307-1.89.012-3.375-.873-4.415-.971-1.142-2.694-1.746-4.983-1.746m.789 6.405c-.373 2.454-2.248 2.454-4.062 2.454h-1.031l.725-4.583a.57.57 0 0 1 .562-.481h.473c1.234 0 2.4 0 3.002.704.359.42.468 1.044.331 1.906m19.653-.079h-3.273a.57.57 0 0 0-.562.481l-.145.916-.23-.332c-.709-1.029-2.289-1.373-3.867-1.373-3.619 0-6.709 2.741-7.311 6.586-.312 1.918.131 3.752 1.219 5.031 1 1.176 2.426 1.666 4.125 1.666 2.916 0 4.533-1.875 4.533-1.875l-.146.91a.57.57 0 0 0 .564.66h2.949a.95.95 0 0 0 .938-.803l1.771-11.209a.57.57 0 0 0-.565-.658m-4.565 6.374c-.314 1.871-1.801 3.127-3.695 3.127-.949 0-1.711-.305-2.199-.883-.484-.574-.666-1.391-.514-2.301.297-1.855 1.805-3.152 3.67-3.152.93 0 1.686.309 2.184.892.501.589.699 1.411.554 2.317m8.426-12.219-2.807 17.858a.57.57 0 0 0 .562.658h2.822c.469 0 .867-.34.939-.803l2.768-17.536a.57.57 0 0 0-.562-.659h-3.16a.57.57 0 0 0-.562.482\"></path><path fill=\"#253B80\" d=\"m7.266 29.154.523-3.322-1.165-.027H1.061L4.927 1.292a.316.316 0 0 1 .314-.268h9.38c3.114 0 5.263.648 6.385 1.927.526.6.861 1.227 1.023 1.917.17.724.173 1.589.007 2.644l-.012.077v.676l.526.298a3.7 3.7 0 0 1 1.065.812c.45.513.741 1.165.864 1.938.127.795.085 1.741-.123 2.812-.24 1.232-.628 2.305-1.152 3.183a6.55 6.55 0 0 1-1.825 2c-.696.494-1.523.869-2.458 1.109-.906.236-1.939.355-3.072.355h-.73c-.522 0-1.029.188-1.427.525a2.2 2.2 0 0 0-.744 1.328l-.055.299-.924 5.855-.042.215c-.011.068-.03.102-.058.125a.16.16 0 0 1-.096.035z\"></path><path fill=\"#179BD7\" d=\"M23.048 7.667q-.042.268-.096.55c-1.237 6.351-5.469 8.545-10.874 8.545H9.326c-.661 0-1.218.48-1.321 1.132L6.596 26.83l-.399 2.533a.704.704 0 0 0 .695.814h4.881c.578 0 1.069-.42 1.16-.99l.048-.248.919-5.832.059-.32c.09-.572.582-.992 1.16-.992h.73c4.729 0 8.431-1.92 9.513-7.476.452-2.321.218-4.259-.978-5.622a4.7 4.7 0 0 0-1.336-1.03\"></path><path fill=\"#222D65\" d=\"M21.754 7.151a10 10 0 0 0-1.203-.267 15 15 0 0 0-2.426-.177h-7.352a1.17 1.17 0 0 0-1.159.992L8.05 17.605l-.045.289a1.336 1.336 0 0 1 1.321-1.132h2.752c5.405 0 9.637-2.195 10.874-8.545q.055-.282.096-.55a6.6 6.6 0 0 0-1.017-.429 9 9 0 0 0-.277-.087\"></path><path fill=\"#253B80\" d=\"M9.614 7.699a1.17 1.17 0 0 1 1.159-.991h7.352c.871 0 1.684.057 2.426.177a10 10 0 0 1 1.481.353q.547.181 1.017.429c.368-2.347-.003-3.945-1.272-5.392C20.378.682 17.853 0 14.622 0h-9.38c-.66 0-1.223.48-1.325 1.133L.01 25.898a.806.806 0 0 0 .795.932h5.791l1.454-9.225z\"></path></svg></div></button><div hidden=\"\" role=\"region\" aria-labelledby=\"button--:r13:--1\" class=\"css-k008qs\" data-reach-accordion-panel=\"\" data-state=\"collapsed\" id=\"panel--:r13:--1\"><div class=\"css-66imkv\" style=\"height: 0px; opacity: 0;\"><div class=\"css-14mzurw\"><div class=\"css-3nnd9c\"><div class=\"css-190uhut\"><span class=\"css-1solgfs\">Continue below to complete your purchase with PayPal.</span></div></div></div></div></div></div></div></div></fieldset></div><div class=\"css-1av2w0m\"><h3 class=\"css-1by5b08\">Billing Address</h3><div class=\"css-0\" data-reach-accordion=\"\"><div data-testid=\"AccordionRadioItem\" class=\"css-9irz3y\" data-reach-accordion-item=\"\" data-state=\"open\"><button aria-controls=\"panel--:r14:--0\" aria-expanded=\"true\" role=\"radio\" aria-checked=\"true\" aria-label=\"Same as shipping address\" class=\"css-1myo5ch\" data-reach-accordion-button=\"\" data-state=\"open\" id=\"button--:r14:--0\"><svg xmlns=\"http://www.w3.org/2000/svg\" fill=\"currentcolor\" viewBox=\"0 0 24 24\" class=\"css-1lehszp\"><path d=\"M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5m0-5C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2m0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8\"></path></svg><span class=\"css-67hcju\">Same as shipping address</span></button><div role=\"region\" aria-labelledby=\"button--:r14:--0\" class=\"css-k008qs\" data-reach-accordion-panel=\"\" data-state=\"open\" id=\"panel--:r14:--0\"><div class=\"css-1wmsx5l\" style=\"height: auto; opacity: 1;\"><div class=\"css-1q4z5sm\"><div class=\"css-1tk1sao\"><p class=\"css-k7yn0b\">25 Sir Williams Lane</p><p class=\"css-k7yn0b\">Etobicoke, ON M9A 1T9 Canada</p></div></div></div></div></div><div data-testid=\"AccordionRadioItem\" class=\"css-1s0xsns\" data-reach-accordion-item=\"\" data-state=\"collapsed\"><button aria-controls=\"panel--:r14:--1\" aria-expanded=\"false\" role=\"radio\" aria-checked=\"false\" aria-label=\"Use a different billing address\" class=\"css-1myo5ch\" data-reach-accordion-button=\"\" data-state=\"collapsed\" id=\"button--:r14:--1\"><svg xmlns=\"http://www.w3.org/2000/svg\" fill=\"currentcolor\" viewBox=\"0 0 24 24\" class=\"css-1s9i1vx\"><path d=\"M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2m0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8\"></path></svg><span class=\"css-67hcju\">Use a different billing address</span></button><div hidden=\"\" role=\"region\" aria-labelledby=\"button--:r14:--1\" class=\"css-k008qs\" data-reach-accordion-panel=\"\" data-state=\"collapsed\" id=\"panel--:r14:--1\"><div class=\"css-14yec2c\" style=\"height: 0px; opacity: 0;\"><div class=\"css-14mzurw\"><div class=\"css-bikomc\"><div class=\"css-zdrrq2\"><label for=\":r15:\" class=\"css-6al9ap\"><input type=\"text\" id=\":r15:\" name=\"firstName\" placeholder=\" \" aria-describedby=\":r15:_error_text\" required=\"\" class=\"css-1xs0zyc\" value=\"\"><div data-role=\"label\" data-testid=\"text-field-label\" class=\"css-1q7zk1e\">First name*</div></label></div><div class=\"css-zdrrq2\"><label for=\":r16:\" class=\"css-6al9ap\"><input type=\"text\" id=\":r16:\" name=\"lastName\" placeholder=\" \" aria-describedby=\":r16:_error_text\" required=\"\" class=\"css-1xs0zyc\" value=\"\"><div data-role=\"label\" data-testid=\"text-field-label\" class=\"css-1q7zk1e\">Last name*</div></label></div><div class=\"css-15owl46\"><div data-testid=\"react-loqate\"><div class=\" css-15owl46\"><label for=\":r17:\" class=\"css-6al9ap\"><input role=\"combobox\" aria-owns=\"loqate-suggestion-list\" aria-controls=\"loqate-suggestion-list\" aria-expanded=\"false\" aria-autocomplete=\"both\" aria-haspopup=\"true\" aria-activedescendant=\"\" data-testid=\"react-loqate-input\" id=\":r17:\" name=\"loqate-address\" placeholder=\" \" aria-describedby=\":r17:_error_text\" class=\"css-1xs0zyc\" value=\"\"><div data-role=\"label\" data-testid=\"text-field-label\" class=\"css-1q7zk1e\">Address</div></label></div><small id=\"loqate-address-error-text\" aria-live=\"polite\" role=\"alert\" class=\"css-xcum31\">Address is required</small><div id=\"loqate-suggestion-list\" role=\"listbox\" aria-live=\"polite\" tabindex=\"0\" hidden=\"\" class=\"css-1c2k4ja\" data-testid=\"react-loqate-list\" style=\"position: relative;\"><div class=\"css-1r6nqzg\"><span class=\"css-olpa1z\">Choose a suggested address to fill address fields or </span>Continue typing address to display results</div><div class=\"css-6ealu\"><button data-testid=\"btn-enter-manual-address-list\" type=\"button\" role=\"option\" aria-selected=\"false\" class=\"css-12endip\"><span class=\"css-a7sbs4\"><span class=\"css-v5mvny\">Manually enter your address</span></span></button></div></div></div></div><button data-testid=\"btn-enter-manual-address-auto-complete\" type=\"button\" class=\"css-tkk1yz\">Manually enter your address</button></div></div></div></div></div></div></div><div class=\"css-y0k0h\"><div class=\"css-1mi13u2\" role=\"alert\"><div class=\"css-1j1lge6\"><div class=\"css-bj81cv\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" fill=\"none\" viewBox=\"0 0 20 20\"><path fill=\"currentColor\" d=\"M10 5.25a.75.75 0 0 1 .75.75v5a.75.75 0 0 1-1.5 0V6a.75.75 0 0 1 .75-.75m1 8.25a1 1 0 1 1-2 0 1 1 0 0 1 2 0\"></path><path fill=\"currentColor\" fill-rule=\"evenodd\" d=\"M20 10c0 5.523-4.477 10-10 10S0 15.523 0 10 4.477 0 10 0s10 4.477 10 10m-1.5 0a8.5 8.5 0 1 1-17 0 8.5 8.5 0 0 1 17 0\" clip-rule=\"evenodd\"></path></svg></div><div class=\"css-872wwp\"><p tabindex=\"-1\" class=\"css-aq9foy\">Transaction processing failed. Please try again with a different payment method.</p></div></div></div></div><div id=\"checkout-confirm-tos-alert\" class=\"css-vurnku\"></div><div class=\"css-1av2w0m\"><h2 class=\"css-1jkqrpx\">30-Day Money Back Guarantee</h2></div><div class=\"css-y0k0h\"><label class=\"css-wm1zje\"><div class=\"css-wy5hsv\"><div class=\"css-x3crln\"><input type=\"checkbox\" name=\"acceptMarketing\" class=\"css-p19g2b\" checked=\"\"><div aria-hidden=\"true\" class=\"css-1va0vdt\"><svg width=\"14\" height=\"11\" viewBox=\"0 0 14 11\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\" class=\"css-8zu6xl\"><path d=\"M13 1 4.692 9.308 1 5.615\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"></path></svg></div></div></div><div class=\"css-bo7t0p\"><span class=\"css-mr7sh\">Get emails with special offers, exclusive products and grooming tips.</span></div></label><label class=\"css-4yeklj\"><input type=\"checkbox\" name=\"acceptMarketing\" aria-label=\"[object Object]\" class=\"css-p19g2b\" checked=\"\"><div class=\"css-1n5gmhe\"><div class=\"css-vurnku\"></div></div><span class=\"css-mr7sh\">Get emails with special offers, exclusive products and grooming tips.</span></label></div><p class=\"css-jen3e4\">By placing your order, you are accepting our <a href=\"/pages/terms-of-use\" target=\"_blank\" rel=\"noreferrer noopener\" class=\"css-1mt3fsu\">Terms of Use and Sale</a>, <a href=\"/pages/product-offer-warranty-return\" target=\"_blank\" rel=\"noreferrer noopener\" class=\"css-1mt3fsu\">Warranty,</a> and you understand that your Peak Hygiene Plan and/or Replenishment Plan will be for an ongoing service that is billed based on the shipping frequency and/or shipping date chosen by you. The recurring charge may change if you change your membership or we change our prices (with notice to you). Cancel anytime by visiting your <a href=\"/account/peak-hygiene-plan\" target=\"_blank\" rel=\"noreferrer noopener\" class=\"css-1mt3fsu\">account</a> page on our website.</p></div><div class=\"css-ozwtzk\"><button aria-label=\"Place my order\" type=\"submit\" form=\"checkoutBillingForm\" data-testid=\"place-order-desktop\" class=\"css-7r3yut\"><span class=\"css-a7sbs4\"><span class=\"css-v5mvny\">Place My Order</span></span></button></div></form>"""
    embeddings = tokenize_html(content[:max_len])

    print(embeddings.shape)
    # to 
    confidence_score = model(embeddings)
    print(confidence_score)