if __name__ == "__main__":
    dataset = CifarDataset("path/to/cifar-10")

    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=0
    )
    for i, batch in enumerate(dataloader):
        img_data, label = batch
        print("image: ", img_data.shape, "label: ", label)
