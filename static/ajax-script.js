let avatar;
$(document).on('submit', '#animate-form', function (e) {
    console.log('hello');
    e.preventDefault();
    $.ajax({
        type: 'POST',
        url: 'change_avatar',
        data: {
            animate: 1,
            avatar: avatar
        },
        success: function (result) {
        }
    })
});

const buttons = document.querySelectorAll('.button');

buttons.forEach(elem => {

    elem.addEventListener("click", (e) => {

        avatar = elem.childNodes[1].attributes.src.value;
        buttons.forEach(function (e) {
            e.classList.remove('active');
        });
        elem.classList.add('active')
        console.log(avatar)
    })

})
