from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
import os
import click

key = os.environ['OPENAI_API_KEY']


@click.command(help='''Uses the file in INPUT_FOLDER to create context
                     for chatGPT, takes questions from user and passes
                     them to chatGPT.''')
@click.argument('input_folder', type=click.Path(exists=True))
@click.option('--recreate_index', '-r',
              is_flag=True, show_default=True,
              default=False,
              help='recreate the index from scratch')
def main(input_folder, recreate_index):
    question = ' '
    if recreate_index or not os.path.exists('index.json'):
        loader = SimpleDirectoryReader(input_folder)
        documents = loader.load_data()
        index = GPTSimpleVectorIndex(documents)
        index.save_to_disk('index.json')
        click.echo(
            f'Successfully processed {len(os.listdir(input_folder))} files.')
    else:
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while question:
        question = input("Type a question :")
        if question != '':
            response = index.query(question)
            print(response)
        else:
            break


if __name__ == "__main__":
    main()
